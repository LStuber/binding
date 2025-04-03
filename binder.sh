#!/bin/bash

# This is a binding script that properly restricts each MPI rank to an appropriate set of CPU cores, GPUs, and NIcs, in order to avoid contention.
#   Usage: mpirun binder.sh <app>
# The exact binding per-rank will be printed at runtime.
# This script can also be used to bind more than 1 MPI rank per GPU, or restrict number of cores and GPUS and perform certain experiments, by setting env variables. Here are the most useful ones:
#   MPI_PER_GPU=2 mpirun -n 8 binder.sh <app>      # bind 2 ranks per GPU
#   LS_SKIP=2 mpirun -n 4 binder.sh <app>          # skip half of the GPUs (eg. 0,2,4,6). Can be useful to understand the impact of different PCI-e topologies, or assign more cores per GPU.
#   OMP_NUM_THREADS=2 mpirun -n 1 binder.sh <app>  # will restrict to 2 cores per rank, using taskset
# The script will auto-enable MPS if multiple ranks are bound to the same GPU.
# The script works by calculating the number of MPI ranks per node, the desired number of ranks per GPU (env variable MPI_PER_GPU), then
# splits resources equally and assigns them to MPI ranks in round robin.
# The script assumes a straightforward CPU - GPU topology (GPU 0 = CPU 0 etc.), and might fail on other topologies, such as EPYC CPUs (although some code is included for them).
# Cores are bound using numactl (optionally taskset to prevent memory binding), GPUs using CUDA_VISIBLE_DEVICES, and NICs using UCX_NET_DEVICES.
# LICENSE : MIT - Copyright Louis Stuber/NVIDIA

set -o pipefail

if [[ -z $1 ]]; then
   echo "Usage: mpirun $0 <app>"
   echo "  will bind your <app> properly and report the binding for each rank (GPUs, CPUs, NICs)."
   echo
   echo "This script reads the following environment variables:"
   echo "  MPI_PER_GPU # default: 1. Desired number of ranks per GPU"
   exit 0
fi


# Get MPI rank/size using SLURM and/or OpenMPI env variables.
if [[ -z $OMPI_COMM_WORLD_LOCAL_RANK ]]; then
   export OMPI_COMM_WORLD_RANK=$SLURM_PROCID
   export OMPI_COMM_WORLD_LOCAL_RANK=$SLURM_LOCALID
   export OMPI_COMM_WORLD_SIZE=$SLURM_NTASKS
   export OMPI_COMM_WORLD_LOCAL_SIZE=$((OMPI_COMM_WORLD_SIZE/SLURM_NNODES))
fi

if [[ -z $OMPI_COMM_WORLD_LOCAL_RANK ]]; then
   export OMPI_COMM_WORLD_RANK=$PMI_RANK
   export OMPI_COMM_WORLD_LOCAL_RANK=$MPI_LOCALRANKID
   export OMPI_COMM_WORLD_SIZE=$PMI_SIZE
   export OMPI_COMM_WORLD_LOCAL_SIZE=$MPI_LOCALNRANKS
fi

if [[ -z $OMPI_COMM_WORLD_LOCAL_RANK ]]; then
   echo "$0 Error: OMPI_COMM_WORLD_LOCAL_RANK not defined. This script only supports SLURM and/or OpenMPI." >&2
   exit 101
fi

if [[ -z $OMPI_COMM_WORLD_LOCAL_SIZE ]]; then
   if [[ -n $LSUB_NTASKS_PER_NODE ]]; then
      OMPI_COMM_WORLD_LOCAL_SIZE=$LSUB_NTASKS_PER_NODE
   else
      echo "$0 error: OMPI_COMM_WORLD_LOCAL_SIZE not defined." >&2
      exit 102
   fi
fi

has_set_ngpus=true
# Detect hardware
# Number of GPUs on the node
if [[ -z $LS_NGPUS ]]; then
   LS_NGPUS=$(nvidia-smi --query-gpu=count --format=csv -i 0 | head -2 | tail -1) || (echo "$0 Error: could not obtain number of GPUs" >&2; exit 103)
   has_set_ngpus=false
fi
if [[ $LS_NGPUS -le 0 ]]; then
   LS_NGPUS=$OMPI_COMM_WORLD_LOCAL_SIZE
fi

# Number of cores and sockets (NUMA nodes are used as "sockets" as they are more reliable)
lscpu="$(lscpu)" #cache
nphys_sockets=${nphys_sockets:-$(echo "$lscpu" | grep Sock | awk '{print $2}')}
if ! [[ "$nphys_sockets" =~ ^[0-9]+$ ]]; then
   echo "$0 error: couldn't detect number of sockets (got '$nphys_sockets'). Please export nphys_sockets=<nsockets>" >&2
   exit 118
fi
ncores=$(echo "$lscpu" | grep "Core(s)" | awk '{print $4}')
nnumas=$(echo "$lscpu" | grep "NUMA node"  | head -1 | awk '{print $3}')
if [[ -n $(echo "$lscpu" | grep "NUMA node$nphys_sockets" | awk '{print $4}') ]]; then
   #System with multiple NUMA nodes per socket (eg. AMD EPYC)
   nsockets=$nnumas
else
   # 1 socket = 1 NUMA
   nnumas=$nphys_sockets
   nsockets=$nphys_sockets
fi

ncores_avail_per_socket=$((ncores*nphys_sockets/nnumas))
if [[ $LS_HYPERTHREAD == true ]] || [[ $LS_HYPERTHREAD == 1 ]]; then
   hyperthread_cores=2
   nthreads_avail_per_socket=$((ncores_avail_per_socket*hyperthread_cores))
else
   hyperthread_cores=1
   nthreads_avail_per_socket=$ncores_avail_per_socket
fi

if [[ -n $NCPU_SOCKETS ]]; then
   nsockets=$NCPU_SOCKETS
elif [[ -n $LS_FORCE_NSOCKETS ]]; then
   nsockets=$LS_FORCE_NSOCKETS
fi


# List the CPUs, NICs on the system, sorted by affinity with GPUs.
isocket=
if ! [[ "$DISABLE_AMD_OPTI" == true ]] && echo "$lscpu" | grep -q "AMD EPYC 7402" && [[ $nsockets == 8 ]] && [[ $LS_NGPUS == 4 ]]; then
   #Special optimization for JUWELS
   GPUS=(0 1 2 3)
   CPUS=(3 2 1 0 7 6 5 4)
   NICS=(mlx5_0 mlx5_1 mlx5_2 mlx5_3)

   if [[ $DISABLE_HALF_NUMAS == true ]]; then
   CPUS=(3 1 7 5)
   fi
   if [[ -n $NICS_PER_NODE ]]; then
      for igpu in $(seq 0 $((LS_NGPUS-1))); do
         NICS[$igpu]=${NICS[$((igpu/(8/NICS_PER_NODE)*(8/NICS_PER_NODE)))]}
      done
   fi
elif ! [[ "$DISABLE_AMD_OPTI" == true ]] && echo "$lscpu" | grep -q "AMD EPYC 7742" && [[ $nsockets == 8 ]] ; then
   #Special optimization for DGX A100
   GPUS=(0 1 2 3 4 5 6 7)
   CPUS=(3 2 1 0 7 6 5 4)
   NICS=(mlx5_1 mlx5_0 mlx5_3 mlx5_2 mlx5_7 mlx5_6 mlx5_9 mlx5_8)

   if [[ $DISABLE_HALF_NUMAS == true ]]; then
   CPUS=(3 1 7 5)
   fi
   if [[ -n $NICS_PER_NODE ]]; then
      for igpu in $(seq 0 $((LS_NGPUS-1))); do
         NICS[$igpu]=${NICS[$((igpu/(8/NICS_PER_NODE)*(8/NICS_PER_NODE)))]}
      done
   fi
else
   #generic system, sequential order assumed
   GPUS=
   CPUS=
   NICS=
   declare -A GPUS CPUS NICS
   if ibstat_out="$(ibstat -l 2>/dev/null | sort -V)" ; then
      mapfile -t ibdevs <<< "${ibstat_out}"
   else
      if [[ -z $NB_NICS ]]; then
         NB_NICS=$(ls -l /dev/infiniband/uverbs* 2>/dev/null | wc -l)
      fi
      ibdevs=
      declare -A ibdevs
      for inic in $(seq 0 $((NB_NICS-1))); do
         ibdevs[$inic]=mlx5_$inic
      done
   fi
   num_ibdevs="${#ibdevs[@]}"
   for igpu in $(seq 0 $((LS_NGPUS-1))); do
      GPUS[$igpu]=$igpu
      NICS[$igpu]=${ibdevs[$((igpu*num_ibdevs/LS_NGPUS))]}

      if [[ -n $NB_NICS_PER_NODE ]]; then
         NICS[$igpu]=${ibdevs[$((igpu*num_ibdevs/LS_NGPUS/(num_ibdevs/NB_NICS_PER_NODE)*(num_ibdevs/NB_NICS_PER_NODE)))]}
      fi
   done

   if ! [[ "$DISABLE_AMD_OPTI" == true ]] && echo "$lscpu" | grep -q "AMD EPYC 77" && [[ $nsockets -gt 1 ]]; then
      unset CPUS
      if  [[ $nsockets -ge 8 ]]; then
         CPUS=(3 2 1 0 7 6 5 4)
      elif  [[ $nsockets -ge 4 ]]; then
         CPUS=(3 2 1 0)
      elif  [[ $nsockets -ge 2 ]]; then
         CPUS=(1 0)
      fi
   else
      if [[ $DISABLE_HALF_NUMAS != true ]]; then
         for icpu in $(seq 0 $((nsockets-1))); do
            CPUS[$icpu]=$((icpu))
         done
      else
         for icpu in $(seq 0 2 $((nsockets-1))); do
            CPUS[$((icpu/2))]=$((icpu))
         done
      fi
   fi
fi

# BINDING

nGPUs=${#GPUS[@]}
nCPUs=${#CPUS[@]}
nNICs=${#NICS[@]}

if [[ $LS_DEBUG == 1 ]] && [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
   if [[ "$nsockets" == "$nCPUs" ]]; then
       nsockets_string="$nsockets"
   else
       nsockets_string="$nsockets ($nCPUs used)"
   fi
   echo "Debug Hardware found: nsockets=$nsockets_string GPUs: ${GPUS[*]} CPUS: ${CPUS[*]}"
fi

# Handle certain special cases

# Detect if multiple MPI ranks are bound to the same GPU and check that it is consistent with MPI_SIZE.
if [[ -z $MPI_PER_GPU ]]; then
   if [[ $nGPUs -lt $OMPI_COMM_WORLD_LOCAL_SIZE ]]; then
      if [[ $OMPI_COMM_WORLD_RANK == 0 ]] || [[ $LS_DEBUG == 1 ]]; then
         echo "Error: there are $OMPI_COMM_WORLD_LOCAL_SIZE MPI ranks per node, but you seem to have only $nGPUs GPUs." >&2
         echo "To prevent mistakes, if you intended to use more than 1 MPI rank per GPU, this script requires you to set the env variable MPI_PER_GPU." >&2
         echo "Try to rerun with: export MPI_PER_GPU=$(((OMPI_COMM_WORLD_LOCAL_SIZE+nGPUs-1)/nGPUs))" >&2
      else
         sleep 2
      fi
      [[ $OMPI_COMM_WORLD_RANK == 1 ]] && echo "$0 error" >&2
      exit 104
   fi
   MPI_PER_GPU=1
else
   if [[ $((nGPUs*MPI_PER_GPU)) -lt $OMPI_COMM_WORLD_LOCAL_SIZE ]]; then
      if [[ $OMPI_COMM_WORLD_RANK == 0 ]] || [[ $LS_DEBUG == 1 ]]; then
         echo "Error: unsatisfiable value for MPI_PER_GPU. $OMPI_COMM_WORLD_LOCAL_SIZE ranks per node spawned, but only $nGPUs GPUs x $MPI_PER_GPU ranks requested." >&2
         echo "Try setting MPI_PER_GPU=$(((OMPI_COMM_WORLD_LOCAL_SIZE+nGPUs-1)/nGPUs))" >&2
      else
         sleep 2
      fi
      [[ $OMPI_COMM_WORLD_RANK == 1 ]] && echo "$0 error" >&2
      exit 105
   fi
fi

if [[ -n "${LS_SKIP}" ]]; then
   export LS_PCI="${LS_SKIP}"
fi

# LS_PCI / LS_SKIP is an experimental flag to skip half the GPUs. Default LS_PCI=1 uses all GPUs. LS_PCI=2 will use 0,2,4,6
if [[ -n $LS_PCI ]]; then
   if [[ $((nGPUs*MPI_PER_GPU)) -lt $((OMPI_COMM_WORLD_LOCAL_SIZE*LS_PCI)) ]]; then
      if [[ $OMPI_COMM_WORLD_RANK == 0 ]] || [[ $LS_DEBUG == 1 ]]; then
         echo "Error: unsatisfiable value for LS_PCI. $OMPI_COMM_WORLD_LOCAL_SIZE ranks per node spawned, but only $((nGPUs/LS_PCI)) GPUs x $MPI_PER_GPU ranks requested." >&2
      else
         sleep 2
      fi
      [[ $OMPI_COMM_WORLD_RANK == 1 ]] && echo "$0 error" >&2
      exit 106
   fi
   ## skip half of the GPUs
   for igpu in $(seq 0 $((nGPUs/LS_PCI-1))); do
      GPUS[$igpu]=${GPUS[$((igpu*LS_PCI))]}
   done
   for igpu in $(seq $((nGPUs/LS_PCI)) $nGPUs); do
      unset "GPUS[$igpu]"
   done
   nGPUs=$((nGPUs/LS_PCI))
   if [[ $nGPUs != ${#GPUS[@]} ]]; then
      echo "Internal $0 error. Please report this bug." >&2
      exit 107
   fi
   if [[ $LS_DEBUG == 1 ]] && [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
      echo "Debug LS_PCI GPUs: ${GPUS[*]} CPUS: ${CPUS[*]}" >&2
   fi
   LS_PCI_DEBUG_STRING=" (some GPUs are hidden due to LS_PCI)"
else
   LS_PCI=1
   LS_PCI_DEBUG_STRING=""
fi

## BINDING
my_gpu_id=$((OMPI_COMM_WORLD_LOCAL_RANK/MPI_PER_GPU))
my_cpu_id=$((OMPI_COMM_WORLD_LOCAL_RANK*nCPUs/nGPUs/MPI_PER_GPU))
my_nic_id=$((OMPI_COMM_WORLD_LOCAL_RANK*nNICs/nGPUs/MPI_PER_GPU))

nmpi_per_node=$((MPI_PER_GPU*nGPUs)) # Max number of ranks that could be spawned if the node was full. More reliable for calculations than OMPI_LOCAL_SIZE which can sometimes differ between nodes.

# GPU binding using CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=${GPUS[$my_gpu_id]}

## NIC binding
export OMPI_MCA_btl_openib_if_include=${NICS[$my_nic_id]}
export UCX_NET_DEVICES=$OMPI_MCA_btl_openib_if_include:1

## CPU binding
# This one is trickier since 2 cases must be handled: multiple NUMA nodes per GPU, or multiple GPUs per NUMA node.

# Note: here we assume that all CPUs must be used and are different, if some NUMA nodes have been disabled due to proximity, that must have been done in earlier steps
nsockets_per_mpi=1
nmpi_per_socket=$((nmpi_per_node/nCPUs))

if [[ $nmpi_per_socket == 0 ]]; then
   #Weird case: multiple sockets are available per MPI rank
   nsockets_per_mpi=$((nCPUs/nGPUs/MPI_PER_GPU))
   nmpi_per_socket=1
fi

requested_env_variable="SCRIPT_AUTODETECTION]"
if [[ -z $NCORES_PER_MPI ]]; then
   if [[ -z $OMP_NUM_THREADS ]]; then
      nthreads_per_mpi=$((nthreads_avail_per_socket/nmpi_per_socket))
   else
      requested_env_variable="OMP_NUM_THREADS"
      nthreads_per_mpi=$OMP_NUM_THREADS
   fi
else
   requested_env_variable="NCORES_PER_MPI"
   nthreads_per_mpi=$NCORES_PER_MPI
fi
if [[ $nsockets_per_mpi -gt 1 ]]; then
   nthreads_avail_per_mpi=$((nthreads_avail_per_socket*nsockets_per_mpi))
else
   nthreads_avail_per_mpi=$((nthreads_avail_per_socket/nmpi_per_socket))
fi
if [[ $nthreads_avail_per_mpi -le 0 ]]; then
   if ! $has_set_ngpus && [[ $OMPI_COMM_WORLD_SIZE -lt $nmpi_per_socket ]]; then
      if [[ $OMPI_COMM_WORLD_RANK == 0 ]] || [[ $LS_DEBUG == 1 ]]; then
         echo "$0 Error: the requested binding settings would cause overlap if all GPUs were used on the node. This is considered an error as it will likely result in wrong GPU/CPU affinity. If you want to only use a subset of the GPUs available, you can do that with 'export LS_NGPUS=N' (acts as if there were N GPUs on the system) or 'export LS_SKIP=N' (skips N devices, eg. LS_SKIP=2 to use GPUs 0,2,4,...)." >&2
      else
         sleep 2
      fi
      exit 114
   else
      if [[ $OMPI_COMM_WORLD_RANK == 0 ]] || [[ $LS_DEBUG == 1 ]]; then
         echo "$0 error. The number of MPI ranks per socket ($nmpi_per_socket) exceeds cores available ($nthreads_avail_per_socket). This is possibly the result of setting MPI_PER_GPU to a value that would cause overlap." >&2
         if [[ -z $LS_PCI ]] || [[ $LS_PCI == 1 ]]; then
            echo "If you want to only use a subset of the GPUs available, you can do that with 'export LS_NGPUS=N' (acts as if there were N GPUs on the system) or 'export LS_SKIP=N' (skips N devices, eg. LS_SKIP=2 to use GPUs 0,2,4,...)." >&2
         fi
      else
         sleep 2
      fi
      exit 112
   fi
fi

if [[ $nthreads_per_mpi -gt $nthreads_avail_per_mpi ]]; then
   if [[ $OMPI_COMM_WORLD_RANK == 0 ]] || [[ $LS_DEBUG == 1 ]]; then
      echo "$0 Error: oversubscription detected. $nthreads_per_mpi cores per rank requested through $requested_env_variable, but only $nthreads_avail_per_mpi available." >&2
      if [[ $LS_HYPERTHREAD == true ]] || [[ $LS_HYPERTHREAD == 1 ]]; then
         echo "Please set $requested_env_variable=$nthreads_avail_per_mpi" >&2
      else
         echo "If you intended to use hyperthreading, please export LS_HYPERTHREAD=true. Otherwise, set $requested_env_variable=$nthreads_avail_per_mpi" >&2
      fi
   else
      sleep 2
   fi
   exit 108;
fi

if [[ $LS_DEBUG == 1 ]] && [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
   echo "Debug nthreads_avail_per_socket=$nthreads_avail_per_socket nthreads_per_mpi=$nthreads_per_mpi nthreads_avail_per_mpi=$nthreads_avail_per_mpi nsockets_per_mpi=$nsockets_per_mpi  nmpi_per_socket=$nmpi_per_socket"
fi

nthreads_avail_per_socket_per_mpi=$(((nthreads_avail_per_mpi+nsockets_per_mpi-1)/nsockets_per_mpi))
nthreads_per_socket_per_mpi=$(((nthreads_per_mpi+nsockets_per_mpi-1)/nsockets_per_mpi))
n=$nthreads_per_mpi
#finally do the socket/core binding
cores=
for my_id in $(seq $my_cpu_id $((my_cpu_id+nsockets_per_mpi-1)) ); do
   isocket=${CPUS[$my_id]}
   #establish list of cores
   intrasocket_rank=$((OMPI_COMM_WORLD_LOCAL_RANK % nmpi_per_socket))
   hyperthread_start=$((ncores_avail_per_socket*nsockets))
   ncores_per_mpi=$((nthreads_per_socket_per_mpi/hyperthread_cores))
   if [[ $ncores_per_mpi == 0 ]]; then
      ncores_per_mpi=1
   fi
   if [[ $CORE_ORDER == sequential ]]; then
      first=$((isocket*ncores_avail_per_socket + intrasocket_rank*nthreads_per_socket_per_mpi))
      ihyper=$(((isocket+1)*ncores_avail_per_socket))
      for i in $(seq 0 $((nthreads_per_socket_per_mpi-1))); do
         if [[ $((first+i)) -lt $ihyper ]]; then
            if [[ $n -gt 0 ]]; then
               cores=$cores$((first+i)),
               n=$((n-1))
            fi
         else
            if [[ $n -gt 0 ]]; then
               #hyperthreading
               cores=$cores$((hyperthread_start+i+first-ihyper+isocket*ncores_avail_per_socket)),
               n=$((n-1))
            fi
         fi
      done
   else
      #Scaled over the range of cores available
      first=$((isocket*ncores_avail_per_socket + intrasocket_rank*nthreads_avail_per_socket_per_mpi))
      ihyper=$(((isocket+1)*ncores_avail_per_socket))
      for i in $(seq 0 $((nthreads_per_socket_per_mpi-1))); do
         idest=$((first+i*nthreads_avail_per_socket_per_mpi/nthreads_per_socket_per_mpi))
         if [[ $idest -lt $ihyper ]]; then
            if [[ $n -gt 0 ]]; then
               cores=$cores$idest,
               n=$((n-1))
            fi
         else
            if [[ $n -gt 0 ]]; then
               #hyperthreading
               cores=$cores$((hyperthread_start+idest-ihyper+isocket*ncores_avail_per_socket)),
               n=$((n-1))
            fi
         fi
      done
   fi
done
#remove last comma
cores=${cores::-1}

if [[ -n $OMP_NUM_THREADS ]] && [[ -z $OMP_PROC_BIND ]]; then
   export OMP_PROC_BIND=true
fi

if [[ -n "$FORCE_UCX_NET_DEVICES" ]]; then
   if [[ "$FORCE_UCX_NET_DEVICES" == unset ]]; then
      unset UCX_NET_DEVICES
   else
      export UCX_NET_DEVICES="$FORCE_UCX_NET_DEVICES"
   fi
fi

if [[ $GPU_BINDING == unset ]]; then
   unset CUDA_VISIBLE_DEVICES
fi

if [[ $GOMP_PIN == 1 ]]; then
   export GOMP_CPU_AFFINITY=$cores
fi

if [[ -z $ENABLE_MPS ]]; then
   if [[ -z $MPI_PER_GPU ]] || [[ $MPI_PER_GPU == 1 ]]; then
      export ENABLE_MPS=false
   else
      export ENABLE_MPS=true
   fi
fi

if [[ $ENABLE_MPS == true ]] || [[ $ENABLE_MPS == 1 ]]; then
   if [[ $OMPI_COMM_WORLD_LOCAL_RANK == 0 ]];then
      env -u CUDA_VISIBLE_DEVICES nvidia-cuda-mps-control -d && echo "Starting MPS on $(hostname)"
   else
      # wait until MPS is started
      timeout 10 bash -c 'until pgrep nvidia-cuda-mps >/dev/null; do sleep 0.5; done' || echo "WARNING: $0 Timed out waiting for the MPS daemon to start (most likely no GPU on the node). Consider disabling MPS to reduce startup overhead (ENABLE_MPS=false)."
   fi
fi

USE_NUMACTL=${USE_NUMACTL:-true}
USE_MEMORY_BINDING=${USE_MEMORY_BINDING:-true}
ADD_MEM_BINDING=""
MEMLIST="<off>"
if $USE_NUMACTL && $USE_MEMORY_BINDING; then
   MEMLIST="${CPUS[$my_cpu_id]}"
   for my_id in $(seq $((my_cpu_id+1)) $((my_cpu_id+nsockets_per_mpi-1)) ); do
      MEMLIST="$MEMLIST,${CPUS[$my_id]}"
   done
   ADD_MEM_BINDING="--membind=$MEMLIST"
fi
export HIP_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

echo "MPI Rank $OMPI_COMM_WORLD_RANK host $(hostname) GPU=$CUDA_VISIBLE_DEVICES cores=$cores membind=$MEMLIST UCX_NET_DEVICES=$UCX_NET_DEVICES"

if $USE_NUMACTL; then
   exec numactl -a --physcpubind=$cores $ADD_MEM_BINDING $@
else
   exec taskset -c $cores $@
fi


if [[ $ENABLE_MPS == true ]] || [[ $ENABLE_MPS == 1 ]]; then
   if [[ $OMPI_COMM_WORLD_LOCAL_RANK == 0 ]];then
      echo quit | nvidia-cuda-mps-control && echo "Stopped MPS"
   fi
fi
