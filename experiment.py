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

print "\n\n----------Stepping through an episode----------"
#We could also start over and do another experiment */
taskSpec = RLGlue.RL_init()

#We could run one step at a time instead of one episode at a time */
#Start the episode */
startResponse = RLGlue.RL_start()

print startResponse.__dict__.keys()

#firstObservation = startResponse.o.intArray[0]
#firstAction = startResponse.a.intArray[0]

stepResponse = RLGlue.RL_step()
#print type(stepResponse.a)
#print type(stepResponse.o)
#print type(stepResponse.r)
print stepResponse.a.intArray.shape
print stepResponse.o.intArray.shape
print stepResponse.r

#Run until the episode ends*/
while (stepResponse.terminal != 1):
    stepResponse = RLGlue.RL_step()
    print stepResponse.r

print "\n\n----------Summary----------"

totalSteps = RLGlue.RL_num_steps()
totalReward = RLGlue.RL_return()
print "It ran for " + str(totalSteps) + " steps, total reward was: " + str(totalReward)
RLGlue.RL_cleanup()
