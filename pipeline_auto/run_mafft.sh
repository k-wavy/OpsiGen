#!/bin/bash
mafft --add $1 --keeplength $2 > $3 # {self.seq_path} {self.alignment_path} {ALIGNED_RESULT_PATH}




