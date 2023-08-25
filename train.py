import sys
sys.path.append('thirdparty/vidar/')

import fire
import thirdparty.vidar.scripts.run as run

if __name__ == '__main__':
    fire.Fire(run.train)