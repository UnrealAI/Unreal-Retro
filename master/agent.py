import argparse
import retro
import pickle

# parser.add_argument('game', help='the name or path for the game to run')
# parser.add_argument('state', nargs='?', help='the initial state file to load, minus the extension')


game, state, record, scenario, players  =  ('Airstriker-Genesis', 'Level1.state', False, 'scenario', 1)
verbose, quiet = 1, 0
env = retro.make(game, state, scenario=scenario, record=record)
print("env: {}".format(dir(env)))
verbosity = verbose - quiet
frame_step = 100

# img_feed = []
class Image_Collector():
    def __init__(self,set_on=False):
        self.SET_ON = set_on
        self.data = []
    def add_img(self,img):
        if(self.SET_ON):
            self.data.append(img)
    def pickle_img(self):
        if(self.SET_ON):
            pickle.dump(self.data, open('image_data.p','wb'))


#debug
img_feed = Image_Collector(set_on=False)
print("env.action_space {}\n".format(env.action_space))
try:
    while True:
        ob = env.reset()
        print("img {}".format(env.img))
        t = 0
        totrew = [0] * players
        while True:
            ac = env.action_space.sample()
            ob, rew, done, info = env.step(ac)
            t += 1
            if t % frame_step == 0:
                if verbosity > 1:
                    infostr = ''
                    if info:
                        infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
                    print(('t=%i' % t) + infostr)
                    img_feed.add_img(env.img)
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
                img_feed.pickle_img()
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
