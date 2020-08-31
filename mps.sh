#!/bin/bash

if [[ $MPI_PER_GPU -gt 1 ]]; then
  if [[ $OMPI_COMM_WORLD_LOCAL_RANK == 0 ]];then
     nvidia-cuda-mps-control -d && echo "Starting MPS on $(hostname)"
  else
     sleep 3
  fi
fi

$@

if [[ $MPI_PER_GPU -gt 1 ]] && [[ $OMPI_COMM_WORLD_LOCAL_RANK == 0 ]];then
echo quit | nvidia-cuda-mps-control && echo "Stopped MPS"
fi

