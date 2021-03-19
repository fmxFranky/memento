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

# download pretrained rainbow agent's model
mkdir -p $3/memento/$1/original_agent/checkpoints/
gsutil cp -r gs://download-dopamine-rl/lucid/rainbow/$1/1/* $3/memento/$1/original_agent/checkpoints/

# train original agent only 1 iteration
python train_original_agent.py \
  --gin_files=./configs/rainbow_199.gin \
  --base_dir=$3/memento/$1/original_agent/ \
  --gin_bindings "atari_lib.create_atari_environment.game_name='$1'" \
  --gin_bindings "Runner.evaluation_steps=$2"

# train memento agent
python train_memento_agent.py \
  --gin_files=./configs/memento.gin \
  --base_dir=$3/memento/$1/memento_agent \
  --original_base_dir=$3/memento/$1/original_agent/ \
  --gin_bindings "atari_lib.create_atari_environment.game_name='$1'" \
  --gin_bindings atari_lib.maybe_transform_variable_names.legacy_checkpoint_load=True
