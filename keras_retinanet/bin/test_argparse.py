import argparse

ap = argparse.ArgumentParser()

######################### 测试 dest 的默认值 ##########################
# ap.add_argument('--imagenet-weights', action='store_const', const=True, default=True)
# dest 默认为 imagenet_weights

######################### 测试 mutual exclusive ######################
group = ap.add_mutually_exclusive_group()
# 只能指定其中一个选项
group.add_argument('--snapshot', help='Resume training from a snapshot.')
# dest 默认是 imagenet_weights
# store_const 表示设置的值为常量
# const 就是常量的值
# default=True 表示即使没有指定 --imagenet-weights, imagenet_weights=True
group.add_argument('--imagenet-weights',
                   help='Initialize the model with pretrained imagenet weights. This is the default behaviour.',
                   action='store_const', const=True, default=True)
group.add_argument('--weights', help='Initialize the model with weights from a file.')
group.add_argument('--no-weights', help='Don not initialize the model with any weights.', dest='imagenet_weights',
                   action='store_const', const=False)
# 什么也不指定的情况 Namespace(imagenet_weights=True, snapshot=None, weights=None)
# 指定 --no-weights Namespace(imagenet_weights=False, snapshot=None, weights=None)
# NOTE: --no-weights 和 --imagenet-weights 使用同一个 dest 并没有影响

print(ap.parse_args())
