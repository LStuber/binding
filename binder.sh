#!/bin/bash

# This is a binding script that was used to restrict number of cores and GPUS
# We do not believe it is of particular interest, but we provide it as requested

if [[ -z $LS_NGPUS ]]; then
   LS_NGPUS=$(nvidia-smi --query-gpu=count --format=csv -i 0 | head -2 | tail -1)
fi

if [[ $LS_NGPUS -ne 8 ]]; then
   echo "WARNING: Number of GPUs is different than 8. count=$LS_NGPUS"
fi

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
ncores_per_socket=$(lscpu | grep "Core(s)" | awk '{print $4}')
nsockets=$(lscpu | grep Sock | awk '{print $2}')

if [[ -n $NCPU_SOCKETS ]]; then
   nsockets=$NCPU_SOCKETS
fi
if [[ -n $LS_FORCE_NSOCKETS ]]; then
   nsockets=$LS_FORCE_NSOCKETS
fi


# calculate cores/sockets per mpi
nmpi_per_socket=$((OMPI_COMM_WORLD_LOCAL_SIZE / nsockets))
if [[ $nmpi_per_socket == 0 ]]; then
   nmpi_per_socket=1
fi
isocket=$((OMPI_COMM_WORLD_LOCAL_RANK / nmpi_per_socket))

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

if [[ $LS_DEBUG == 1 ]]; then
   echo "Debug ncores_per_socket=$ncores_per_socket nsockets=$nsockets LS_FORCE_NSOCKETS=$LS_FORCE_NSOCKETS LS_NCORES_PER_MPI=$LS_NCORES_PER_MPI nmpi_per_socket=$nmpi_per_socket OMP_NUM_THREADS=$OMP_NUM_THREADS ncores_per_mpi=$ncores_per_mpi"
fi

if [[ -n $OMP_NUM_THREADS ]] && [[ -z $OMP_PROC_BIND ]]; then
   export OMP_PROC_BIND=true
fi


#establish list of cores
intrasocket_rank=$((OMPI_COMM_WORLD_LOCAL_RANK % nmpi_per_socket))
first=$((isocket*ncores_per_socket + intrasocket_rank*ncores_per_mpi))
if [[ -n $LS_SOCKET_START ]]; then
first=$((first+LS_SOCKET_START))
fi
cores=$first
for i in $(seq 1 $((ncores_per_mpi-1))); do
   cores=$cores,$((first+i))
done

# Tmp workaround to add Redstone 2nd socket cores to the 1st one
if [[ $LS_REDSTONE_SOCKET == 1 ]]; then
   first=$(((isocket+1)*ncores_per_socket + intrasocket_rank*ncores_per_mpi))
   cores=$cores,$first
   for i in $(seq 1 $((ncores_per_mpi-1))); do
      cores=$cores,$((first+i))
   done
fi

if [[ $HYPERTHREAD == 1 ]]; then
   hyperthread_start=$((ncores_per_socket*nsockets))
   for c in $(echo $cores | tr ',' ' '); do
      cores=$cores,$((c+hyperthread_start))
   done
fi

# GPU binding
if [[ $LS_DISABLE_GPU_BINDING != true ]]; then
export ACC_DEVICE_NUM=$OMPI_COMM_WORLD_LOCAL_RANK
if [[ $LS_DISABLE_GPU_BINDING != onlyacc ]]; then
   export ACC_DEVICE_NUM=0
   if [[ -n $LS_GPUBINDING_NGPUS ]]; then
      export CUDA_VISIBLE_DEVICES=""
      for i in $(seq 0 $((LS_GPUBINDING_NGPUS/nsockets-1))); do
         CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES$((i+isocket*LS_GPUBINDING_NGPUS/nsockets)),"
      done
   else
      export CUDA_VISIBLE_DEVICES=$(((LS_NGPUS/nsockets )*isocket + intrasocket_rank))
   fi
   if [[ $LS_GPU_BINDING_ALL_VISIBLE == 1 ]]; then
      export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | sed "s/$OMPI_COMM_WORLD_LOCAL_RANK,//g")
      CUDA_VISIBLE_DEVICES=$(echo $CUDA_VISIBLE_DEVICES | sed "s/,$OMPI_COMM_WORLD_LOCAL_RANK//g")
      CUDA_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK,$CUDA_VISIBLE_DEVICES
   fi
fi
if [[ -n $MPI_PER_GPU ]]; then
export LS_MPS=$MPI_PER_GPU
fi
if [[ -n $LS_MPS ]] && [[ $LS_MPS -gt 0 ]]; then
   export CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK/LS_MPS))
fi
if [[ -n $LS_PCI ]] && [[ $LS_MPS -gt 0 ]]; then
   if [[ -n $LS_MPS ]]; then
      export CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK/LS_MPS*LS_PCI))
   else
      export CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK*LS_PCI))
   fi
fi
if [[ -n $CVD ]]; then
   export CUDA_VISIBLE_DEVICES=${CVD:$OMPI_COMM_WORLD_LOCAL_RANK:1}
fi
if [[ -n $LS_NVLINK ]]; then
   export CUDA_VISIBLE_DEVICES=$((OMPI_COMM_WORLD_LOCAL_RANK%2+OMPI_COMM_WORLD_LOCAL_RANK/2*4))
fi
fi

if [[ $LS_NIC_BINDING == 1 ]]; then
   nnic=4
   mpis_per_nic=$((nmpi_per_socket*nsockets/nnic))
   export OMPI_MCA_btl_openib_if_include="mlx5_$((OMPI_COMM_WORLD_LOCAL_RANK/mpis_per_nic)):1"
fi
if [[ $LS_QE_TRICK == 1 ]]; then
   if [[ $LS_NCORES_PER_MPI != 4 ]]; then
      echo "Error LS_NCORES_PER_MPI must be 4 for QE trick"
      exit -1
   fi
   if [[ $((OMPI_COMM_WORLD_RANK % 4)) == 0 ]]; then
      export MKL_NUM_THREADS=7
      export OMP_NUM_THREADS=7
      cores=$cores,$((16+OMPI_COMM_WORLD_LOCAL_RANK*5)),$((17+OMPI_COMM_WORLD_LOCAL_RANK*5)),$((18+OMPI_COMM_WORLD_LOCAL_RANK*5))
   fi
fi

if [[ $CP2K_TRICK == 1 ]]; then
  if [[ $OMPI_COMM_WORLD_LOCAL_RANK == 0 ]]; then
    cores=0,1,2,3,4,5,6,7,8,9
    export CUDA_VISIBLE_DEVICES=0
    export UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1"
  fi
  if [[ $OMPI_COMM_WORLD_LOCAL_RANK == 1 ]]; then
    cores=20,21,22,23,24,25,26,27,28,29
    export CUDA_VISIBLE_DEVICES=4
    export UCX_NET_DEVICES="mlx5_2:1,mlx5_3:1"
  fi
  if [[ $OMPI_COMM_WORLD_LOCAL_RANK == 2 ]]; then
    cores=30,31,32,33,34,35,36,37,38,39
    export CUDA_VISIBLE_DEVICES=6
    export UCX_NET_DEVICES="mlx5_2:1,mlx5_3:1"
  fi
  if [[ $OMPI_COMM_WORLD_LOCAL_RANK == 3 ]]; then
    cores=10,11,12,13,14,15,16,17,18,19
    export CUDA_VISIBLE_DEVICES=2
    export UCX_NET_DEVICES="mlx5_0:1,mlx5_1:1"
  fi
fi

echo Rank $OMPI_COMM_WORLD_RANK host $(hostname) GPU $CUDA_VISIBLE_DEVICES cores $cores $OMPI_MCA_btl_openib_if_include $UCX_NET_DEVICES
exec taskset -c $cores $@
