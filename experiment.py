# 
# Copyright (C) 2008, Brian Tanner
# 
#http://rl-glue-ext.googlecode.com/
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
#

import sys
import rlglue.RLGlue as RLGlue

whichEpisode = 0


def runEpisode(stepLimit):
    global whichEpisode
    terminal = RLGlue.RL_episode(stepLimit)
    totalSteps = RLGlue.RL_num_steps()
    totalReward = RLGlue.RL_return()

    print "Episode " + str(whichEpisode) + "\t " + str(totalSteps) + " steps \t" + str(totalReward) + " total reward\t " + str(terminal) + " natural end"

    whichEpisode = whichEpisode+1


print "\n\n----------Stepping through an episode----------"
#We could also start over and do another experiment */
taskSpec = RLGlue.RL_init()

for i in range(50):
    runEpisode(100000)
