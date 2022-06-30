import sys
import os
cpu_affinity = os.sched_getaffinity(0)
import numpy as np

temp = open('eranpath','r')
Eranfolder = temp.readline()[:-1]
sys.path.insert(0, Eranfolder+'/ELINA/python_interface/')
sys.path.insert(0, Eranfolder+'/deepg/code/')

from eran import ERAN
from read_net_file import *
from read_zonotope_file import read_zonotope
import tensorflow as tf
import csv
import time
from tqdm import tqdm
from ai_milp import *
import argparse
from config import config
from constraint_utils import *
import re
import itertools
from multiprocessing import Pool, Value
import onnxruntime.backend as rt
import logging


#ZONOTOPE_EXTENSION = '.zt'
EPS = 10**(-9)

is_tf_version_2=tf.__version__[0]=='2'

if is_tf_version_2:
    tf= tf.compat.v1


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def isnetworkfile(fname):
    _, ext = os.path.splitext(fname)
    if ext not in ['.pyt', '.meta', '.tf','.onnx', '.pb']:
        raise argparse.ArgumentTypeError('only .pyt, .tf, .onnx, .pb, and .meta formats supported')
    return fname



def parse_input_box(text):
    intervals_list = []
    for line in text.split('\n'):
        if line!="":
            interval_strings = re.findall("\[-?\d*\.?\d+, *-?\d*\.?\d+\]", line)
            intervals = []
            for interval in interval_strings:
                interval = interval.replace('[', '')
                interval = interval.replace(']', '')
                [lb,ub] = interval.split(",")
                intervals.append((np.double(lb), np.double(ub)))
            intervals_list.append(intervals)

    # return every combination
    boxes = itertools.product(*intervals_list)
    return list(boxes)


def show_ascii_spec(lb, ub, n_rows, n_cols, n_channels):
    print('==================================================================')
    for i in range(n_rows):
        print('  ', end='')
        for j in range(n_cols):
            print('#' if lb[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ', end='')
        for j in range(n_cols):
            print('#' if ub[n_cols*n_channels*i+j*n_channels] >= 0.5 else ' ', end='')
        print('  |  ')
    print('==================================================================')

'''
def normalize(image, means, stds, dataset):
    # normalization taken out of the network
    if len(means) == len(image):
        for i in range(len(image)):
            image[i] -= means[i]
            if stds!=None:
                image[i] /= stds[i]
    elif dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = (image[i] - means[0])/stds[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = (image[count] - means[0])/stds[0]
            count = count + 1
            tmp[count] = (image[count] - means[1])/stds[1]
            count = count + 1
            tmp[count] = (image[count] - means[2])/stds[2]
            count = count + 1

        if(is_conv):
            for i in range(3072):
                image[i] = tmp[i]
        else:
            count = 0
            for i in range(1024):
                image[i] = tmp[count]
                count = count+1
                image[i+1024] = tmp[count]
                count = count+1
                image[i+2048] = tmp[count]
                count = count+1



def normalize_poly(num_params, lexpr_cst, lexpr_weights, lexpr_dim, uexpr_cst, uexpr_weights, uexpr_dim, means, stds, dataset):
    # normalization taken out of the network
    if dataset == 'mnist' or dataset == 'fashion':
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[0]) / stds[0]
            uexpr_cst[i] = (uexpr_cst[i] - means[0]) / stds[0]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[0]
            uexpr_weights[i] /= stds[0]
    else:
        for i in range(len(lexpr_cst)):
            lexpr_cst[i] = (lexpr_cst[i] - means[i % 3]) / stds[i % 3]
            uexpr_cst[i] = (uexpr_cst[i] - means[i % 3]) / stds[i % 3]
        for i in range(len(lexpr_weights)):
            lexpr_weights[i] /= stds[(i // num_params) % 3]
            uexpr_weights[i] /= stds[(i // num_params) % 3]


def denormalize(image, means, stds, dataset):
    if dataset == 'mnist'  or dataset == 'fashion':
        for i in range(len(image)):
            image[i] = image[i]*stds[0] + means[0]
    elif(dataset=='cifar10'):
        count = 0
        tmp = np.zeros(3072)
        for i in range(1024):
            tmp[count] = image[count]*stds[0] + means[0]
            count = count + 1
            tmp[count] = image[count]*stds[1] + means[1]
            count = count + 1
            tmp[count] = image[count]*stds[2] + means[2]
            count = count + 1

        for i in range(3072):
            image[i] = tmp[i]
'''

def model_predict(base, input):
    if is_onnx:
        pred = base.run(input)
    else:
        pred = base.run(base.graph.get_operation_by_name(model.op.name), {base.graph.get_operations()[0].name + ':0': input})
    return pred


def estimate_grads(specLB, specUB, dim_samples=3):
    specLB = np.array(specLB, dtype=np.float32)
    specUB = np.array(specUB, dtype=np.float32)
    inputs = [((dim_samples - i) * specLB + i * specUB) / dim_samples for i in range(dim_samples + 1)]
    diffs = np.zeros(len(specLB))

    # refactor this out of this method
    if is_onnx:
        runnable = rt.prepare(model, 'CPU')
    elif sess is None:
        runnable = tf.Session()
    else:
        runnable = sess

    for sample in range(dim_samples + 1):
        pred = model_predict(runnable, inputs[sample])

        for index in range(len(specLB)):
            if sample < dim_samples:
                l_input = [m if i != index else u for i, m, u in zip(range(len(specLB)), inputs[sample], inputs[sample+1])]
                l_input = np.array(l_input, dtype=np.float32)
                l_i_pred = model_predict(runnable, l_input)
            else:
                l_i_pred = pred
            if sample > 0:
                u_input = [m if i != index else l for i, m, l in zip(range(len(specLB)), inputs[sample], inputs[sample-1])]
                u_input = np.array(u_input, dtype=np.float32)
                u_i_pred = model_predict(runnable, u_input)
            else:
                u_i_pred = pred
            diff = np.sum([abs(li - m) + abs(ui - m) for li, m, ui in zip(l_i_pred, pred, u_i_pred)])
            diffs[index] += diff
    return diffs / dim_samples



progress = 0.0
def print_progress(depth):
    if config.debug:
        global progress, rec_start
        progress += np.power(2.,-depth)
        sys.stdout.write('\r%.10f percent, %.02f s' % (100 * progress, time.time()-rec_start))

'''
def acasxu_recursive(specLB, specUB, max_depth=10, depth=0):
    hold,nn,nlb,nub,_,_ = eran.analyze_box(specLB, specUB, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
    global failed_already
    if hold:
        print_progress(depth)
        return hold
    elif depth >= max_depth:
        if failed_already.value and config.complete:
            verified_flag, adv_examples = verify_network_with_milp(nn, specLB, specUB, nlb, nub, constraints)
            print_progress(depth)
            if verified_flag == False:
                if adv_examples!=None:
                    #print("adv image ", adv_image)
                    for adv_image in adv_examples:
                        hold,_,nlb,nub,_,_ = eran.analyze_box(adv_image, adv_image, domain, config.timeout_lp, config.timeout_milp, config.use_default_heuristic, constraints)
                        #print("hold ", hold, "domain", domain)
                        if hold == False:
                            print("property violated at ", adv_image, "output_score", nlb[-1])
                            failed_already.value = 0
                            break
            return verified_flag
        else:
            return False
    else:
        grads = estimate_grads(specLB, specUB)
        # grads + small epsilon so if gradient estimation becomes 0 it will divide the biggest interval.
        smears = np.multiply(grads + 0.00001, [u-l for u, l in zip(specUB, specLB)])

        #start = time.time()
        #nn.set_last_weights(constraints)
        #grads_lower, grads_upper = nn.back_propagate_gradiant(nlb, nub)
        #smears = [max(-grad_l, grad_u) * (u-l) for grad_l, grad_u, l, u in zip(grads_lower, grads_upper, specLB, specUB)]
        index = np.argmax(smears)
        m = (specLB[index]+specUB[index])/2

        result =  failed_already.value and acasxu_recursive(specLB, [ub if i != index else m for i, ub in enumerate(specUB)], max_depth, depth + 1)
        result = failed_already.value and result and acasxu_recursive([lb if i != index else m for i, lb in enumerate(specLB)], specUB, max_depth, depth + 1)
        return result



def get_tests(dataset, geometric):
    if geometric:
        csvfile = open(Eranfolder+'/deepg/code/datasets/{}_test.csv'.format(dataset), 'r')
    else:
        if config.subset == None:
            csvfile = open(Eranfolder+'/data/{}_test.csv'.format(dataset), 'r')
        else:
            filename = Eranfolder+'/data/'+ dataset+ '_test_' + config.subset + '.csv'
            csvfile = open(filename, 'r')
    tests = csv.reader(csvfile, delimiter=',')

    return tests
'''

def init_domain(d):
    if d == 'refinezono':
        return 'deepzono'
    elif d == 'refinepoly':
        return 'deeppoly'
    else:
        return d

parser = argparse.ArgumentParser(description='ERAN Example',  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--netname', type=isnetworkfile, default=config.netname, help='the network name, the extension can be only .pb, .pyt, .tf, .meta, and .onnx')
parser.add_argument('--epsilon', type=float, default=config.epsilon, help='the epsilon for L_infinity perturbation')
parser.add_argument('--zonotope', type=str, default=config.zonotope, help='file to specify the zonotope matrix')
parser.add_argument('--subset', type=str, default=config.subset, help='suffix of the file to specify the subset of the test dataset to use')
parser.add_argument('--target', type=str, default=config.target, help='file specify the targets for the attack')
parser.add_argument('--epsfile', type=str, default=config.epsfile, help='file specify the epsilons for the L_oo attack')
#parser.add_argument('--specnumber', type=int, default=config.specnumber, help='the property number for the acasxu networks')
parser.add_argument('--domain', type=str, default=config.domain, help='the domain name can be either deepzono, refinezono, deeppoly or refinepoly')
#parser.add_argument('--dataset', type=str, default=config.dataset, help='the dataset, can be either mnist, cifar10, acasxu, or fashion')
parser.add_argument('--complete', type=str2bool, default=config.complete,  help='flag specifying where to use complete verification or not')
parser.add_argument('--timeout_lp', type=float, default=config.timeout_lp,  help='timeout for the LP solver')
parser.add_argument('--timeout_milp', type=float, default=config.timeout_milp,  help='timeout for the MILP solver')
parser.add_argument('--numproc', type=int, default=config.numproc,  help='number of processes for MILP / LP / k-ReLU')
parser.add_argument('--sparse_n', type=int, default=config.sparse_n,  help='Number of variables to group by k-ReLU')
parser.add_argument('--use_default_heuristic', type=str2bool, default=config.use_default_heuristic,  help='whether to use the area heuristic for the DeepPoly ReLU approximation or to always create new noise symbols per relu for the DeepZono ReLU approximation')
parser.add_argument('--use_milp', type=str2bool, default=config.use_milp,  help='whether to use milp or not')
parser.add_argument('--refine_neurons', action='store_true', default=config.refine_neurons, help='whether to refine intermediate neurons')
parser.add_argument('--mean', nargs='+', type=float, default=config.mean, help='the mean used to normalize the data with')
parser.add_argument('--std', nargs='+', type=float, default=config.std, help='the standard deviation used to normalize the data with')
parser.add_argument('--data_dir', type=str, default=config.data_dir, help='data location')
parser.add_argument('--geometric_config', type=str, default=config.geometric_config, help='config location')
parser.add_argument('--num_params', type=int, default=config.num_params, help='Number of transformation parameters')
parser.add_argument('--num_tests', type=int, default=config.num_tests, help='Number of images to test')
parser.add_argument('--from_test', type=int, default=config.from_test, help='Number of images to test')
parser.add_argument('--debug', action='store_true', default=config.debug, help='Whether to display debug info')
parser.add_argument('--attack', action='store_true', default=config.attack, help='Whether to attack')
parser.add_argument('--geometric', '-g', dest='geometric', default=config.geometric, action='store_true', help='Whether to do geometric analysis')
parser.add_argument('--input_box', default=config.input_box,  help='input box to use')
parser.add_argument('--output_constraints', default=config.output_constraints, help='custom output constraints to check')
parser.add_argument('--normalized_region', type=str2bool, default=config.normalized_region, help='Whether to normalize the adversarial region')

# added by nikos for experiments
parser.add_argument('--sanity_check', type=str2bool, default=config.sanity_check, help='perform sanity check')
parser.add_argument('--splitting',type=str2bool,default=config.splitting, help='Use splitting for polynomials')
parser.add_argument('--lower_bound',type=float,default=config.lower_bound, help='Choose lower bound for NN inputs')
parser.add_argument('--upper_bound',type=float,default=config.upper_bound, help='Choose upper bound for NN inputs')
parser.add_argument('--poly_dynamic',type=str2bool,default=config.poly_dynamic,help="Use hard-coded polynomial or dynamic")
parser.add_argument('--poly_order',type=int,default=config.poly_order,help="choose polynomial order for tansig approx.")
parser.add_argument('--num_inputs',type=int,default=config.num_inputs,help="number of inputs.")

# Logging options
parser.add_argument('--logdir', type=str, default=None, help='Location to save logs to. If not specified, logs are not saved and emitted to stdout')
parser.add_argument('--logname', type=str, default=None, help='Directory of log files in `logdir`, if not specified timestamp is used')



#Parse and save all the input options
args = parser.parse_args()
for k, v in vars(args).items():
    setattr(config, k, v)
config.json = vars(args)
'''
if config.specnumber and not config.input_box and not config.output_constraints:
    config.input_box = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_input_prenormalized.txt'
    config.output_constraints = '../data/acasxu/specs/acasxu_prop_' + str(config.specnumber) + '_constraints.txt'

assert config.netname, 'a network has to be provided for analysis.'
'''
#if len(sys.argv) < 4 or len(sys.argv) > 5:
#    print('usage: python3.6 netname epsilon domain dataset')
#    exit(1)

netname = config.netname
filename, file_extension = os.path.splitext(netname)

is_trained_with_pytorch = file_extension==".pyt"
is_saved_tf_model = file_extension==".meta"
is_pb_file = file_extension==".pb"
is_tensorflow = file_extension== ".tf"
is_onnx = file_extension == ".onnx"
assert is_trained_with_pytorch or is_saved_tf_model or is_pb_file or is_tensorflow or is_onnx, "file extension not supported"

epsilon = config.epsilon
#assert (epsilon >= 0) and (epsilon <= 1), "epsilon can only be between 0 and 1"

zonotope_file = config.zonotope
zonotope = None
zonotope_bool = (zonotope_file!=None)
if zonotope_bool:
    zonotope = read_zonotope(zonotope_file)

domain = config.domain

if zonotope_bool:
    assert domain in ['deepzono', 'refinezono'], "domain name can be either deepzono or refinezono"
elif not config.geometric:
    assert domain in ['deepzono', 'refinezono', 'deeppoly', 'refinepoly'], "domain name can be either deepzono, refinezono, deeppoly or refinepoly"

dataset = config.dataset
'''
if zonotope_bool==False:
   assert dataset in ['mnist', 'cifar10', 'acasxu', 'fashion'], "only mnist, cifar10, acasxu, and fashion datasets are supported"
'''
constraints = None
if config.output_constraints:
    constraints = get_constraints_from_file(config.output_constraints)

mean = 0
std = 0

complete = (config.complete==True)
'''
if(dataset=='acasxu'):
    print("netname ", netname, " specnumber ", config.specnumber, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)
else:
    print("netname ", netname, " epsilon ", epsilon, " domain ", domain, " dataset ", dataset, "args complete ", config.complete, " complete ",complete, " timeout_lp ",config.timeout_lp)
'''
non_layer_operation_types = ['NoOp', 'Assign', 'Const', 'RestoreV2', 'SaveV2', 'PlaceholderWithDefault', 'IsVariableInitialized', 'Placeholder', 'Identity']

sess = None
if is_saved_tf_model or is_pb_file:
    netfolder = os.path.dirname(netname)

    tf.logging.set_verbosity(tf.logging.ERROR)

    sess = tf.Session()
    if is_saved_tf_model:
        saver = tf.train.import_meta_graph(netname)
        saver.restore(sess, tf.train.latest_checkpoint(netfolder+'/'))
    else:
        with tf.gfile.GFile(netname, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.graph_util.import_graph_def(graph_def, name='')
    ops = sess.graph.get_operations()
    last_layer_index = -1
    while ops[last_layer_index].type in non_layer_operation_types:
        last_layer_index -= 1
    model = sess.graph.get_tensor_by_name(ops[last_layer_index].name + ':0')
    eran = ERAN(model, sess)

else:
    if(zonotope_bool==True):
        num_pixels = len(zonotope)
    elif(dataset=='mnist'):
        num_pixels = 784
    elif (dataset=='cifar10'):
        num_pixels = 3072
    elif(dataset=='acasxu'):
        num_pixels = 5
    if is_onnx:
        model, is_conv = read_onnx_net(netname)
    else:
        num_pixels = 784
        if np.isnan(config.num_inputs):
            num_inputs=num_pixels
        else:
            num_inputs=config.num_inputs
        model, is_conv, means, stds = read_tensorflow_net(netname, num_inputs, is_trained_with_pytorch, (domain == 'gpupoly' or domain == 'refinegpupoly'))
    eran = ERAN(model, is_onnx=is_onnx)
'''
if not is_trained_with_pytorch:
    if dataset == 'mnist' and not config.geometric:
        means = [0]
        stds = [1]
    elif dataset == 'acasxu':
        means = [1.9791091e+04,0.0,0.0,650.0,600.0]
        stds = [60261.0,6.28318530718,6.28318530718,1100.0,1200.0]
    else:
        means = [0.5, 0.5, 0.5]
        stds = [1, 1, 1]
'''
is_trained_with_pytorch = is_trained_with_pytorch or is_onnx

if config.mean is not None:
    means = config.mean
    stds = config.std

os.sched_setaffinity(0,cpu_affinity)

correctly_classified_images = 0
verified_images = 0

#Take the tests from the selected dataset
if dataset:
    if config.input_box is None:
        tests = get_tests(dataset, config.geometric)
    else:
        tests = open(config.input_box, 'r').read()

def init(args):
    global failed_already
    failed_already = args


num_inputs = 784
specLB = config.lower_bound*np.ones(num_inputs, dtype="double")
specUB = config.upper_bound*np.ones(num_inputs, dtype="double")
'''
for i in range(len(specLB)):
    specLB[i]=-2*np.random.randn(1)-3
    specUB[i]=1*np.random.randn(1)+4
print(specLB)
print(specUB)
'''

start_sapo = time.time()
# Come step 1
# eran is the ERAN object defined l.425 in __main__.py
# analyze_box is an ERAN class function defined l.59 in eran.py
label,nn,nlb_SAPO,nub_SAPO,_,_ = eran.analyze_box(specLB, specUB, 'refinepoly', config.timeout_lp, config.timeout_milp, config.use_default_heuristic)#label=label, prop=prop)
end_sapo = time.time()

# print("The total time ERAN is ", end-start, "seconds")
# print("lower bounds _eran:{}".format(nlb_ERAN[-1]))
# print("upper bounds _eran:{}".format(nub_ERAN[-1]))
print("The total time ERAN+SAPO is ", end_sapo-start_sapo, "seconds")
print("lower bounds _sapo:{}".format(nlb_SAPO[-1]))
print("upper bounds _sapo:{}".format(nub_SAPO[-1]))
