# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
set -x

# bash run.sh Game num_eval_steps output_dir
game=$1
eval_steps=$2
output_dir=$(cd $(dirname $3); pwd)/$game

# download pretrained rainbow agent's model
mkdir -p $output_dir/original_agent/checkpoints/
gsutil cp -r gs://download-dopamine-rl/lucid/rainbow/$game/1/* $output_dir/original_agent/checkpoints/

# train original agent only 1 iteration
python train_original_agent.py \
  --gin_files=./configs/rainbow_199.gin \
  --base_dir=$output_dir/original_agent/ \
  --gin_bindings "atari_lib.create_atari_environment.game_name='$game'" \
  --gin_bindings "Runner.evaluation_steps=$eval_steps"

# train memento agent
python train_memento_agent.py \
  --gin_files=./configs/memento.gin \
  --base_dir=$output_dir/memento_agent \
  --original_base_dir=$output_dir/original_agent/ \
  --gin_bindings "atari_lib.create_atari_environment.game_name='$game'" \
  --gin_bindings atari_lib.maybe_transform_variable_names.legacy_checkpoint_load=True
