rl_glue > rl_glue.log 2>&1  &
ale -game_controller rlglue ~/ale_0.4.3/ale_0_4/roms/breakout.bin > ale.log 2>&1 &
python $HEDGEHOG/scripts/experiment.py > experiment.log 2>&1 &
THEANO_FLAGS="floatX=float32,device=gpu" python $HEDGEHOG/agents/basic.py > agent.log 2>&1 &
