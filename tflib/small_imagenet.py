import numpy as np
import scipy.misc
import time
import os
def make_generator(path, n_files, batch_size, img_size):
    epoch_count = [1]
    img_list = os.listdir(path)
    def get_epoch():
        images = np.zeros((batch_size, 3, img_size, img_size), dtype='int32')
        files = list(range(n_files))
        random_state = np.random.RandomState(epoch_count[0])
        # print(files)
        random_state.shuffle(files)
        # print("Not Again!!!!!!!!!!")
        epoch_count[0] += 1
        for n, i in enumerate(files):
            # image = scipy.misc.imread("{}{}.png".format(path, str(i+1).zfill(7)))
            image = scipy.misc.imread("{}{}".format(path, img_list[i]))
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            images[n % batch_size] = image.transpose(2,0,1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size, data_dir='/content/improved_wgan_training/imagenet_64/', img_size=64):
    return (
        make_generator(data_dir+'train_'+str(img_size)+'/', 431, batch_size, img_size),
        make_generator(data_dir+'val_'+str(img_size)+'/', 49, batch_size, img_size)
        # make_generator(data_dir+'train_64x64/train_64x64/', 6880, batch_size),
        # make_generator(data_dir+'val_64x64/val_64x64/', 800, batch_size)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print ("{} {}".format(str(time.time() - t0), batch[0][0,0,0,0]))
        if i == 1000:
            break
        t0 = time.time()