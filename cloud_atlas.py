'''
Based off the code found in 
tensorflow/python/models/tutorials/image/cifar10/
and the associated file structure.

I copy many of those functions over verbatim and modify the code to my needs.

I modify the architecture of the CNN here to my needs and to the size of the data set.
'''

import tensorflow as tf

import cloud_atlas_input

tf.app.flags.DEFINE_integer('batch_size', 128, '''number of images to process in a batch''')
tf.app.flags.DEFINE_string('data_dir', './data', '''Path to categorized cloud images''')
tf.app.flags.DEFINE_boolean('use_fp16', './data', '''Train the model using floating point 16 vars''')

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

# verbatim from tf code
def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    :param  x: Tensor
    :return: nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                         tf.nn.zero_fraction(x))


def inputs(eval_data):
    '''Construct input for evaluation on cloud types using Reader ops.

    :param eval_data: bool
        Whether to use the train or eval dat set
    :return: images, labels
        images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
        labels: Labels. 1D tensor of [batch_size] size.
    :raise:
        ValueError: If no data_dir
    '''
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cloud_atlas-batches-bin')
    images, labels = cloud_atlas_input.inputs(eval_data=eval_data,
                                              data_dir=data_dir,
                                              batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def inference(images):
    '''Build the model for a cloud atlas.

    :param images: images returned from distorted_inputs() or inputs()
    :return: list
        Logits for given images
    '''

    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 3, 64], # TODO(dstone): check if needs modified
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME') # TODO: what is [1,1,1,1] array?
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1') # TODO: understand these parameters

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 64, 64], # TODO(dstone): check if needs modified
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME') # TODO: what is [1,1,1,1] array?
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    # TODO(dstone): copy over rest of model from tensorflow_models/tutorials/images/cifar10/cifar10.py

