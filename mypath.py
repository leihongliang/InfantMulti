import os


class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'infant_multi_labels':
            # folder that contains class labels
            # root_dir = '/infant'
            root_dir = '/student1/hl_lei/InfantMulti/infant'

            # 128*171 112
            output_dir = '/student1/hl_lei/InfantMulti/video2pic'
            # 256*342 224


            # output_dir = '/student1/hl_lei/InfantMulti/video2pic2'


            return root_dir, output_dir
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def model_dir():
        # return '/pretrained/c3d-pretrained.pth'
        return '/student1/hl_lei/InfantMulti/pretrained/ucf101-caffe.pth'



