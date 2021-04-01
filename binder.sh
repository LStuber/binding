#!/bin/bash

# This is a binding script that was used to restrict number of cores and GPUS
# We do not believe it is of particular interest, but we provide it as requested

set -o pipefail

if [[ -z $LS_NGPUS ]]; then
   LS_NGPUS=$(nvidia-smi --query-gpu=count --format=csv -i 0 | head -2 | tail -1) || (echo "Could not obtain number of GPUs"; exit -1)
fi

##if [[ $LS_NGPUS -ne 8 ]]; then
##   echo "WARNING: Number of GPUs is different than 8. count=$LS_NGPUS"
##fi

if [[ -z $OMPI_COMM_WORLD_LOCAL_SIZE ]]; then
   if [[ -n $LSUB_NTASKS_PER_NODE ]]; then
      OMPI_COMM_WORLD_LOCAL_SIZE=$LSUB_NTASKS_PER_NODE
   else
      echo "error OMPI_COMM_WORLD_LOCAL_SIZE not defined"
      exit -1
   fi
fi
if [[ -z $OMPI_COMM_WORLD_LOCAL_RANK ]]; then
   export OMPI_COMM_WORLD_RANK=$SLURM_PROCID
   export OMPI_COMM_WORLD_LOCAL_RANK=$SLURM_LOCALID
   export OMPI_COMM_WORLD_SIZE=$SLURM_NTASKS
   export OMPI_COMM_WORLD_LOCAL_SIZE=$((OMPI_COMM_WORLD_SIZE/SLURM_NNODES))
fi

if [[ -z $OMPI_COMM_WORLD_LOCAL_RANK ]]; then
   echo "Error: OMPI_COMM_WORLD_LOCAL_RANK not defined"
   exit -1
fi


#hardware
#do not touch, required for cores count
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

# GPU binding
   if [[ -n $MPI_PER_GPU ]]; then
      export LS_MPS=$MPI_PER_GPU
   fi
   nmpi_per_gpu=1
   if [[ -n $LS_MPS ]]; then
      nmpi_per_gpu=$((nmpi_per_gpu*LS_MPS))
   fi
   if [[ $((nmpi_per_gpu*LS_NGPUS)) -lt $OMPI_COMM_WORLD_LOCAL_SIZE ]]; then
      echo "Error: too many MPI ranks per GPU. $nmpi_per_gpu requested, $(((OMPI_COMM_WORLD_LOCAL_SIZE+LS_NGPUS-1)/LS_NGPUS)) found. Try setting MPI_PER_GPU=$(((OMPI_COMM_WORLD_LOCAL_SIZE+LS_NGPUS-1)/LS_NGPUS))"
      exit -1
   fi
   export CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK/nmpi_per_gpu))
   if [[ -n $LS_PCI ]]; then
      if [[ $((nmpi_per_gpu*LS_NGPUS/LS_PCI)) -lt $OMPI_COMM_WORLD_LOCAL_SIZE ]]; then
         echo "Error: too many MPI ranks per GPU. $nmpi_per_gpu requested, $((OMPI_COMM_WORLD_LOCAL_SIZE/LS_NGPUS)) found. Try tweaking MPI_PER_GPU and LS_PCI."
         exit -1
      fi
      if [[ $LS_PCI == 2ALLNUMA ]]; then
         export CUDA_VISIBLE_DEVICES_actual=$((CUDA_VISIBLE_DEVICES/2*2))
      else
         #since LS_MPS is "nb of MPIs per GPU", but LS_PCI cheats, the actual LS_MPS desired by the user is LS_MPS/LS_PCI
         export CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK*LS_PCI/nmpi_per_gpu))
         export CUDA_VISIBLE_DEVICES_actual=$((OMPI_COMM_WORLD_LOCAL_RANK/nmpi_per_gpu*LS_PCI))
      fi
   fi

isocket=
if ! [[ "$DISABLE_AMD_OPTI" == true ]] && lscpu | grep -q "AMD EPYC 7402" && [[ LS_NGPUS == 4 ]]; then
   #JUWELS
   GPUS=(0 1 2 3)
   CPUS=(3 4 1 2 7 8 5 6)
   NICS=(mlx5_0 mlx5_1 mlx5_2 mlx5_3)

   if [[ $DISABLE_HALF_NUMAS == true ]]; then
   CPUS=(3 3 1 1 7 7 5 5)
   fi
   if [[ -n $NICS_PER_NODE ]]; then
      for igpu in $(seq 0 $((LS_NGPUS-1))); do
         NICS[$igpu]=${NICS[$((igpu/(8/NICS_PER_NODE)*(8/NICS_PER_NODE)))]}
      done
   fi
elif lscpu | grep -q "AMD EPYC 7742" && ! [[ "$DISABLE_AMD_OPTI" == true ]]; then
   #DGX A100
   GPUS=(0 1 2 3 4 5 6 7)
   CPUS=(3 2 1 0 7 6 5 4)
   NICS=(mlx5_1 mlx5_0 mlx5_3 mlx5_2 mlx5_7 mlx5_6 mlx5_9 mlx5_8)

   if [[ $DISABLE_HALF_NUMAS == true ]]; then
   CPUS=(3 3 1 1 7 7 5 5)
   fi
   if [[ -n $NICS_PER_NODE ]]; then
      for igpu in $(seq 0 $((LS_NGPUS-1))); do
         NICS[$igpu]=${NICS[$((igpu/(8/NICS_PER_NODE)*(8/NICS_PER_NODE)))]}
      done
   fi
else
   #generic system
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
   readonly num_ibdevs="${#ibdevs[@]}"
   for igpu in $(seq 0 $((LS_NGPUS-1))); do
      GPUS[$igpu]=$igpu
      CPUS[$igpu]=$((igpu*nsockets/LS_NGPUS))
      NICS[$igpu]=${ibdevs[$((igpu*num_ibdevs/LS_NGPUS))]}

      if [[ $DISABLE_HALF_NUMAS == true ]]; then
         CPUS[$igpu]=$((igpu*nsockets/LS_NGPUS/2*2))
      fi
      if [[ -n $NB_NICS_PER_NODE ]]; then
         NICS[$igpu]=${ibdevs[$((igpu*num_ibdevs/LS_NGPUS/(num_ibdevs/NB_NICS_PER_NODE)*(num_ibdevs/NB_NICS_PER_NODE)))]}
      fi
   done
##   echo "Rank $OMPI_COMM_WORLD_RANK $CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES_actual CPUS ${CPUS[0]} ${CPUS[1]} ${CPUS[2]} ${CPUS[3]} ${CPUS[4]} ${CPUS[5]} ${CPUS[6]} ${CPUS[7]} ${CPUS[8]}"
fi

## NIC binding
export OMPI_MCA_btl_openib_if_include=${NICS[${CUDA_VISIBLE_DEVICES::1}]}
export UCX_NET_DEVICES=$OMPI_MCA_btl_openib_if_include:1

## CPU binding

# calculate cores/sockets per mpi
if [[ -z $nmpi_per_socket ]]; then
   nmpi_per_socket=$((LS_NGPUS*nmpi_per_gpu / nsockets))
   ##      nmpi_per_socket=$((OMPI_COMM_WORLD_LOCAL_SIZE / nsockets))
fi
if [[ -n $LS_PCI ]]; then
   nmpi_per_socket=$((nmpi_per_socket/LS_PCI))
fi
if [[ $nmpi_per_socket == 0 ]]; then
   echo "Error: $nmpi_per_socket MPI ranks per socket"
   export nmpi_per_socket=1
   exit -1
fi
##if [[ $DISABLE_HALF_NUMAS == true ]]; then
##   nmpi_per_socket=$((nmpi_per_socket*2))
##fi
if [[ -z $isocket ]]; then
   isocket=${CPUS[$((OMPI_COMM_WORLD_LOCAL_RANK / nmpi_per_socket))]}
fi

if [[ -n $NCORES_PER_MPI ]];then
   export LS_NCORES_PER_MPI=$NCORES_PER_MPI
fi
if [[ -z $LS_NCORES_PER_MPI ]];then
   if [[ -z $OMP_NUM_THREADS ]]; then
      ncores_per_mpi=$((ncores_per_socket/nmpi_per_socket))
   else
      ncores_per_mpi=$OMP_NUM_THREADS
   fi
else
   ncores_per_mpi=$LS_NCORES_PER_MPI
fi

if [[ $ncores_per_mpi -gt $((ncores_per_socket/nmpi_per_socket)) ]]; then
   echo "Error: oversubscription detected. $ncores_per_mpi cores per MPI used, $((ncores_per_socket/nmpi_per_socket)) available"
   exit -1;
fi

if [[ $LS_DEBUG == 1 ]]; then
   echo "Debug ncores_per_socket=$ncores_per_socket nsockets=$nsockets LS_FORCE_NSOCKETS=$LS_FORCE_NSOCKETS LS_NCORES_PER_MPI=$LS_NCORES_PER_MPI nmpi_per_socket=$nmpi_per_socket OMP_NUM_THREADS=$OMP_NUM_THREADS ncores_per_mpi=$ncores_per_mpi tmp_gpuid=$tmp_gpuid isocket=$isocket"
fi

if [[ -n $OMP_NUM_THREADS ]] && [[ -z $OMP_PROC_BIND ]]; then
   export OMP_PROC_BIND=true
fi

if [[ -z $ncores_per_mpi_aligned ]]; then
   ncores_per_mpi_aligned=ncores_per_mpi
fi

if [[ $((ncores_per_socket/nmpi_per_socket )) -lt $ncores_per_mpi_aligned ]]; then
   echo "Error: oversubscription detected. $ncores_per_mpi_aligned cores/MPI requested, $((ncores_per_socket/nmpi_per_socket )) available"
   exit -1
fi

#establish list of cores
intrasocket_rank=$((OMPI_COMM_WORLD_LOCAL_RANK % nmpi_per_socket))
##first=$((isocket*ncores_per_socket + intrasocket_rank*ncores_per_mpi_aligned))
first=$((isocket*ncores_per_socket + intrasocket_rank*ncores_per_socket/nmpi_per_socket))
if [[ -n $LS_SOCKET_START ]]; then
   first=$((first+LS_SOCKET_START))
fi
cores=$first
for i in $(seq 1 $((ncores_per_mpi-1))); do
   cores=$cores,$((first+i))
done

if [[ $HYPERTHREAD == 1 ]]; then
   hyperthread_start=$((ncores_per_socket*nsockets))
   for c in $(echo $cores | tr ',' ' '); do
      cores=$cores,$((c+hyperthread_start))
   done
fi


if [[ -n "$FORCE_UCX_NET_DEVICES" ]]; then
   if [[ "$FORCE_UCX_NET_DEVICES" == unset ]]; then
      unset UCX_NET_DEVICES
   else
      export UCX_NET_DEVICES="$FORCE_UCX_NET_DEVICES"
   fi
fi

if [[ -n $CUDA_VISIBLE_DEVICES_actual ]]; then
   export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_actual
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

echo MPI Rank $OMPI_COMM_WORLD_RANK host $(hostname) GPU=$CUDA_VISIBLE_DEVICES cores=$cores UCX_NET_DEVICES=$UCX_NET_DEVICES
exec taskset -c $cores $@

if [[ $ENABLE_MPS == true ]] || [[ $ENABLE_MPS == 1 ]]; then
   if [[ $OMPI_COMM_WORLD_LOCAL_RANK == 0 ]];then
      echo quit | nvidia-cuda-mps-control && echo "Stopped MPS"
   fi
fi
