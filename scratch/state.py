import argparse
import retro
import pickle

# parser.add_argument('game', help='the name or path for the game to run')
# parser.add_argument('state', nargs='?', help='the initial state file to load, minus the extension')


game, state, record, scenario, players  =  ('Airstriker-Genesis', 'Level1.state', False, 'scenario', 1)
verbose, quiet = 0, 0
env = retro.make(game, state, scenario=scenario, record=record)
print("env: {}".format(dir(env)))
verbosity = verbose - quiet
frame_step = 100

# img_feed = []
# class Image_collector():
#     self.data = []


#debug
print("env.action_space {}\n".format(env.action_space))
try:
    while True:
        ob = env.reset()
        print("img {}".format(env.img))
        # img_feed.append(env.img)
        t = 0
        totrew = [0] * players
        while True:
            ac = env.action_space.sample()
            ob, rew, done, info = env.step(ac)

            t += 1
            if t % frame_step == 0:
                print('IMAGE', ob.shape)
                if verbosity > 1:
                    infostr = ''
                    if info:
                        infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
                    print(('t=%i' % t) + infostr)
                # img_feed.append(env.img)
                env.render()

            if players == 1:
                rew = [rew]
            for i, r in enumerate(rew):
                totrew[i] += r
                if verbosity > 0:
                    if r > 0:
                        print('t=%i p=%i got reward: %g, current reward: %g' % (t, i, r, totrew[i]))
                    if r < 0:
                        print('t=%i p=%i got penalty: %g, current reward: %g' % (t, i, r, totrew[i]))
            if done:
                env.render()
                # pickle.dump(img_feed, open('image_data.p','wb'))
                try:
                    if verbosity >= 0:
                        if players > 1:
                            print("done! total reward: time=%i, reward=%r" % (t, totrew))
                        else:
                            print("done! total reward: time=%i, reward=%d" % (t, totrew[0]))
                        input("press enter to continue")
                        print()
                    else:
                        input("")
                except EOFError:
                    exit(0)
                break
except KeyboardInterrupt:
    exit(0)
