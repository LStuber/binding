#!/bin/bash

# This is a binding script that properly restricts each MPI rank to an appropriate set of CPU cores, GPUs, and NIcs, in order to avoid contention.
#   Usage: mpirun binder.sh <app>
# This script can also be used to restrict number of cores and GPUS and perform certain experiments.
# It will auto-enable MPS if multiple ranks are bound on the same GPU.
# The code basically counts the number of MPI ranks per node, the desired number of ranks per GPU (env variable MPI_PER_GPU), then
# splits resources equally and assigns them to MPI ranks in round robin. An exception is made for certain AMD CPUs where the optimal NUMA affinity is not straighforward.
# Cores are bound using taskset, GPUs using CUDA_VISIBLE_DEVICES, and NICs using UCX_NET_DEVICES.

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
   echo "$0 Error: OMPI_COMM_WORLD_LOCAL_RANK not defined. This script only supports SLURM and/or OpenMPI."
   exit 101
fi

if [[ -z $OMPI_COMM_WORLD_LOCAL_SIZE ]]; then
   if [[ -n $LSUB_NTASKS_PER_NODE ]]; then
      OMPI_COMM_WORLD_LOCAL_SIZE=$LSUB_NTASKS_PER_NODE
   else
      echo "$0 error: OMPI_COMM_WORLD_LOCAL_SIZE not defined."
      exit 102
   fi
fi


# Detect hardware
# Number of GPUs on the node
if [[ -z $LS_NGPUS ]]; then
   LS_NGPUS=$(nvidia-smi --query-gpu=count --format=csv -i 0 | head -2 | tail -1) || (echo "$0 Error: could not obtain number of GPUs"; exit 103)
fi
# Number of cores and sockets (NUMA nodes are used as "sockets" as they are more reliable)
nsockets=$(lscpu | grep Sock | awk '{print $2}')
ncores_per_socket=$(lscpu | grep "Core(s)" | awk '{print $4}')
nnumas=$(lscpu | grep "NUMA node"  | head -1 | awk '{print $3}')

ncores_per_socket=$((ncores_per_socket*nsockets/nnumas))
nsockets=$nnumas

if [[ -n $NCPU_SOCKETS ]]; then
   nsockets=$NCPU_SOCKETS
elif [[ -n $LS_FORCE_NSOCKETS ]]; then
   nsockets=$LS_FORCE_NSOCKETS
fi


# List the CPUs, NICs on the system, sorted by affinity with GPUs.
isocket=
if ! [[ "$DISABLE_AMD_OPTI" == true ]] && lscpu | grep -q "AMD EPYC 7402" && [[ $LS_NGPUS == 4 ]]; then
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
elif lscpu | grep -q "AMD EPYC 7742" && ! [[ "$DISABLE_AMD_OPTI" == true ]]; then
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
         NB_NICS=$(ls /dev/infiniband/uverbs* | wc -l)
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
         echo "Error: there are $OMPI_COMM_WORLD_LOCAL_SIZE MPI ranks per node, but you seem to have only $nGPUs GPUs."
         echo "To prevent mistakes, if you intended to use more than 1 MPI rank per GPU, this script requires you to set the env variable MPI_PER_GPU."
         echo "Try to rerun with: export MPI_PER_GPU=$(((OMPI_COMM_WORLD_LOCAL_SIZE+nGPUs-1)/nGPUs))"
      else
         sleep 2
      fi
      [[ $OMPI_COMM_WORLD_RANK == 1 ]] && echo "$0 error"
      exit 104
   fi
   MPI_PER_GPU=1
else
   if [[ $((nGPUs*MPI_PER_GPU)) -lt $OMPI_COMM_WORLD_LOCAL_SIZE ]]; then
      if [[ $OMPI_COMM_WORLD_RANK == 0 ]] || [[ $LS_DEBUG == 1 ]]; then
         echo "Error: unsatisfiable value for MPI_PER_GPU. $OMPI_COMM_WORLD_LOCAL_SIZE ranks per node spawned, but only $nGPUs GPUs x $MPI_PER_GPU ranks requested."
         echo "Try setting MPI_PER_GPU=$(((OMPI_COMM_WORLD_LOCAL_SIZE+nGPUs-1)/nGPUs))"
      else
         sleep 2
      fi
      [[ $OMPI_COMM_WORLD_RANK == 1 ]] && echo "$0 error"
      exit 105
   fi
fi

# LS_PCI is an experimental flag to skip half the GPUs. Default LS_PCI=1 uses all GPUs. LS_PCI=2 will use 0,2,4,6
if [[ -n $LS_PCI ]]; then
   if [[ $((nGPUs*MPI_PER_GPU)) -lt $((OMPI_COMM_WORLD_LOCAL_SIZE*LS_PCI)) ]]; then
      if [[ $OMPI_COMM_WORLD_RANK == 0 ]] || [[ $LS_DEBUG == 1 ]]; then
         echo "Error: unsatisfiable value for LS_PCI. $OMPI_COMM_WORLD_LOCAL_SIZE ranks per node spawned, but only $((nGPUs/LS_PCI)) GPUs x $MPI_PER_GPU ranks requested."
      else
         sleep 2
      fi
      [[ $OMPI_COMM_WORLD_RANK == 1 ]] && echo "$0 error"
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
      echo "Internal $0 script error. Please report this bug."
      exit 107
   fi
   if [[ $LS_DEBUG == 1 ]] && [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
      echo "Debug LS_PCI GPUs: ${GPUS[*]} CPUS: ${CPUS[*]}"
   fi
   LS_PCI_DEBUG_STRING=" (some GPUs are hidden due to LS_PCI)"
else
   LS_PCI=1
   LS_PCI_DEBUG_STRING=""
fi

# Last special case: on certain CPUs, some NUMA nodes do not have affinity to the GPU (eg. DGX).
# removed?

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
   nsockets_per_mpi=$((nCPUs/nGPUs))
   nmpi_per_socket=1
fi

requested_env_variable="[INTERNAL SCRIPT]"
if [[ -z $NCORES_PER_MPI ]]; then
   if [[ -z $OMP_NUM_THREADS ]]; then
      ncores_per_mpi=$((ncores_per_socket/nmpi_per_socket))
   else
      requested_env_variable="OMP_NUM_THREADS"
      if [[ $LS_HYPERTHREAD == true ]]; then
         ncores_per_mpi=$((OMP_NUM_THREADS/2))
      else
         ncores_per_mpi=$OMP_NUM_THREADS
      fi
   fi
else
   requested_env_variable="NCORES_PER_MPI"
   ncores_per_mpi=$NCORES_PER_MPI
fi
ncores_per_mpi_avail=$((ncores_per_socket/nmpi_per_socket))

if [[ $ncores_per_mpi -gt $ncores_per_mpi_avail ]]; then
   if [[ $OMPI_COMM_WORLD_RANK == 0 ]] || [[ $LS_DEBUG == 1 ]]; then
      echo "Error: oversubscription detected. $ncores_per_mpi cores per rank requested through $requested_env_variable, but only $ncores_per_mpi_avail available."
      if ! [[ $LS_HYPERTHREAD == true ]]; then
         echo "If you intended to use hyperthreading, please export LS_HYPERTHREAD=true. Otherwise, set $requested_env_variable=$ncores_per_mpi_avail"
      fi
   else
      sleep 2
   fi
   exit 108;
fi

if [[ $LS_DEBUG == 1 ]] && [[ $OMPI_COMM_WORLD_RANK == 0 ]]; then
   echo "Debug ncores_per_socket=$ncores_per_socket ncores_per_mpi=$ncores_per_mpi ncores_per_mpi_avail=$ncores_per_mpi_avail nsockets_per_mpi=$nsockets_per_mpi  nmpi_per_socket=$nmpi_per_socket"
fi

#finally do the socket/core binding
for my_id in $(seq $my_cpu_id $((my_cpu_id+nsockets_per_mpi-1)) ); do
   isocket=${CPUS[$my_id]}
   #establish list of cores
   intrasocket_rank=$((OMPI_COMM_WORLD_LOCAL_RANK % nmpi_per_socket))
   if [[ $CORE_ORDER == sequential ]]; then
      first=$((isocket*ncores_per_socket + intrasocket_rank*ncores_per_mpi))
      cores=$first
      for i in $(seq 1 $((ncores_per_mpi-1))); do
         cores=$cores,$((first+i))
      done
   else
      #Scaled over the range of cores available
      first=$((isocket*ncores_per_socket + intrasocket_rank*ncores_per_mpi_avail))
      cores=$first
      for i in $(seq 1 $((ncores_per_mpi-1))); do
         cores=$cores,$((first+i*ncores_per_mpi_avail/ncores_per_mpi))
      done
   fi
   if [[ $LS_HYPERTHREAD == 1 ]]; then
      hyperthread_start=$((ncores_per_socket*nsockets))
      for c in $(echo $cores | tr ',' ' '); do
         cores=$cores,$((c+hyperthread_start))
      done
   fi
done

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
      sleep 0.5
   fi
fi

export HIP_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

echo "MPI Rank $OMPI_COMM_WORLD_RANK host $(hostname) GPU=$CUDA_VISIBLE_DEVICES cores=$cores UCX_NET_DEVICES=$UCX_NET_DEVICES"
exec taskset -c $cores $@

if [[ $ENABLE_MPS == true ]] || [[ $ENABLE_MPS == 1 ]]; then
   if [[ $OMPI_COMM_WORLD_LOCAL_RANK == 0 ]];then
      echo quit | nvidia-cuda-mps-control && echo "Stopped MPS"
   fi
fi
