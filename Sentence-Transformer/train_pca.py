import os
import math
import logging
import argparse
import warnings
from sklearn.decomposition import PCA
import torch

from torch.utils.tensorboard import SummaryWriter
from larva import LarvaTokenizer, LarvaModel
from sentence_transformers import SentenceTransformer, LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from sentence_transformers.pair_data import PairData
import pandas as pd
import glob
import random
import pickle
import numpy as np
import os
import shutil


warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train(gpu, ngpus_per_node, args):
    # https://tutorials.pytorch.kr/intermediate/dist_tuto.html
# def train(args):
    args.gpu = gpu + args.start_gpu_num
    device = torch.device("cuda:{}".format(args.gpu))

    if args.gpu is not None:
        logging.debug("Use GPU: {} for training".format(args.gpu))

    args.rank = args.rank * ngpus_per_node + gpu
    print("gpu :", args.gpu, "rank :", args.rank, "world size", args.world_size)
    dist.init_process_group("nccl", init_method='tcp://127.0.0.1:39501',
                            rank=args.rank, world_size=args.world_size)   
    torch.cuda.set_device(args.gpu)

    logging.debug('Pretrained model loading...')
    # 1. For word embedding and pooling
    model = LarvaModel.from_pretrained(args.larva_model)
    tokenizer = LarvaTokenizer.from_pretrained(args.larva_model)
    model.save_pretrained(args.root_dir + args.pt_model_dir)
    tokenizer.save_pretrained(args.root_dir + args.pt_model_dir)


    logging.debug('Pretrained model loaded!')

    # 2. path
    train_data_path = args.data_dir + "/" + args.train_dir + "/"
    valid_data_path = args.data_dir + '/valid/'
    test_data_path = args.data_dir + '/test/'


    logging.debug('MODEL DESCRIPTIONS')
    # 3. pre trained larva model
    word_embedding_model = models.Transformer(args.root_dir + args.pt_model_dir) # n*3*256
    cnn = models.CNN(
        in_word_embedding_dimension=word_embedding_model.get_word_embedding_dimension(), 
        out_channels=256, 
        kernel_sizes=[1,3,5]
    ) # n*3*768 (256*3)
    pooling_model = models.Pooling(
        cnn.get_word_embedding_dimension(), 
        pooling_mode_mean_tokens=True, 
        pooling_mode_cls_token=False, 
        pooling_mode_max_tokens=False
    )
    # dropout = models.Dropout(0.2)
    # sent_embeddings_dimension = pooling_model.get_sentence_embedding_dimension()
    # dan1 = models.Dense(in_features=sent_embeddings_dimension, out_features=sent_embeddings_dimension,)
    # dan2 = models.Dense(in_features=sent_embeddings_dimension, out_features=args.output_dim)

    # sbert_model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dan1, dan2], device=device)
    sbert_model = SentenceTransformer(modules=[word_embedding_model,cnn, pooling_model], device=device)

    # 4. data loader
    # valid_pair = PairData(valid_data_path)
    # valid_samples = valid_pair.get_example(shuffle=False, num_data=args.valid_size, gpu=args.gpu)


    logging.debug('Validataion dataset is same to test data set : rf_hdfs')

    test_pair = PairData(test_data_path + "rf_hdfs")
    test_samples = test_pair.get_example(shuffle=False, num_data=args.valid_size, gpu=args.gpu)
    train_pair = PairData(train_data_path)
    train_data_iter = train_pair.get_data_iter(args.batch_size, is_train=True, duplicates=args.duplicates, gpu=args.gpu)

    # 5. loss function
    evaluator = BinaryClassificationEvaluator.from_input_examples(test_samples, name=args.data_name)
    evaluation_steps = int(args.train_size/args.batch_size)
    if args.loss_type == "contrastive":
        train_loss = losses.ContrastiveLoss(model=sbert_model)
    elif args.loss_type == "online_contrastive":
        train_loss = losses.OnlineContrastiveLoss(model=sbert_model)
    elif args.loss_type == "cos_sim":
        train_loss = losses.CosineSimilarityLoss(model=sbert_model)
    elif args.loss_type == "on_cont_cross_ent_mt":
        train_loss_1 = losses.OnlineContrastiveLoss(model=sbert_model)
        train_loss_2 = losses.SoftmaxLoss(model=sbert_model, num_labels=2, sentence_embedding_dimension=pooling_model.get_sentence_embedding_dimension())
    elif args.loss_type == "on_cont_cross_ent":
        train_loss = losses.OnlineContrastiveCrossEntropyLoss(
            model=sbert_model,
            num_labels=2, 
            sentence_embedding_dimension=pooling_model.get_sentence_embedding_dimension()
            )
    warmup_steps = math.ceil(args.train_size * args.epochs * args.warmup_rate)

    #5. model 이름
    model_output_name = "".join(args.ymd.split("-")) + "_{}_{}_{}e_{}bs_{}_{}" \
            .format(args.pt_model_dir.split("/")[-1], args.loss_type, args.epochs, args.batch_size, args.train_dir, args.output_dim)

    # 6. tensorboard
    writer = SummaryWriter(args.tensorboard_path + "/" + model_output_name)

    # 7. train
    output_model_dir = args.output_model_dir + '/' + model_output_name ##잔오류 수정입니다. 
    sbert_model.fit(
        train_objectives=[(train_data_iter, train_loss)],
        # train_objectives=[(train_data_iter, train_loss_1), (train_data_iter, train_loss_2)],
        evaluator=evaluator,
        epochs=args.epochs,
        evaluation_steps=evaluation_steps,
        warmup_steps=warmup_steps,
        output_path=args.root_dir + output_model_dir,
        show_progress_bar=False,
        writer=writer, 
        device_ids=[args.gpu], 
        output_device=args.gpu,
        rank=args.rank,
        # optimizer_params={'lr': 2e-5}
        optimizer_params={'lr': 0.0003}
        # optimizer_params={'lr': 0.0005}
    )


def test(args):
    model_output_name = "".join(args.ymd.split("-")) + "_{}_{}_{}e_{}bs_{}_{}" \
            .format(args.pt_model_dir.split("/")[-1], args.loss_type, args.epochs, args.batch_size, args.train_dir, args.output_dim)
    output_model_dir = args.output_model_dir + "/" + model_output_name
    sbert_model = SentenceTransformer(
        args.root_dir + output_model_dir, 
        device="cuda:{}".format(args.gpu)
        )   

    ######################
    ### 모든 train data 읽어들이기. 

    logging.debug('Reading Train data Starts.....')

    train_data_path = '/home1/irteam/sbchoi/workspace/Work/BM/3rd_modelling/data/train' # use your path
    all_files = glob.glob(os.path.join(train_data_path , "*"))

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, delimiter='\t', index_col=None, header=None)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    query_list = frame.iloc[:, 1].tolist() + frame.iloc[:, 2].tolist()

    logging.debug('Reading Train data Done.....')

    ## random하게 10퍼센트로 쿼리 뽑기. 
    random_query = random.sample(query_list, int(len(query_list)*0.1))



    #############################
    # encoding multi-gpu에서 하는것. 

    # emb = sbert_model.encode(query_list, device= 'cuda')  ## gpu 1개만 쓸거면 이렇게 하면됨. 
    #Start the multi-process pool on all available CUDA devices
    logging.debug("Sbert encoding starts....")
    pool = sbert_model.start_multi_process_pool()
    logging.debug("Use all available GPUs for encode")
    #Compute the embeddings using the multi-process pool
    emb = sbert_model.encode_multi_process(random_query, pool)
    logging.debug("Sbert encoding is done")

   
    embed_dir = '/home1/irteam/sbchoi/workspace/Work/BM/3rd_modelling/data/embed/{}'.format(model_output_name)
    new_dimension = 128 ## 조절

    #### pca 오래걸리므로 이미 한 pca는 다시 안하기 위해서..
    if os.path.exists(embed_dir):
        logging.debug("already pca has been done so just loading..")

        with open(embed_dir + '/embdding_vector_pca.npy', 'rb') as f:
            pca_comp = np.load(f) 
    
    else:       

    #########################
        ## PCA
        #Compute PCA on the train embeddings matrix

        logging.debug('Starting PCA TO 128 Dimension.....')

        pca = PCA(n_components=new_dimension)
        pca.fit(emb)
        logging.debug('PCA IS DONE !!!!!')
        pca_comp = np.asarray(pca.components_)
        ##########################

        ## embedding과 sample query list 저장.
    

        if not os.path.exists(embed_dir):
            os.makedirs(embed_dir)
        else:
            shutil.rmtree(embed_dir)
            os.makedirs(embed_dir)


        with open(embed_dir + '/embdding_vector_pca.npy'.format(model_output_name), 'wb') as f:
            np.save(f, pca_comp)

        with open(embed_dir + '/query_sample_list.pkl'.format(model_output_name), 'wb') as file:
            pickle.dump(random_query, file)

        logging.debug('Saving embed and list of query is done!!!!')


    ##########################
    ## pca to dense
    # We add a dense layer to the model, so that it will produce directly embeddings with the new size
    
    dense = models.Dense(in_features=sbert_model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=torch.nn.Identity())
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    sbert_model.add_module('dense', dense)

    sbert_model.save('/home1/irteam/sbchoi/workspace/Work/BM/3rd_modelling/output_model_with_pca/' +model_output_name + '_pca')



    for test_data in ['/test/rf_hdfs', '/test/labeling_1_1', '/test/labeling_1_2' , '/test/quality_test_1']: ## test data목록, 나중에 추가되면 os로 바꾸자. 
  
        test_data_path = args.data_dir + test_data
        test_pair = PairData(test_data_path)
        test_samples = test_pair.get_example(shuffle=False, num_data=args.valid_size, gpu=args.gpu)
        test_evaluator = BinaryClassificationEvaluator.from_input_examples(test_samples, name=args.data_name)
        logging.debug('This is test prediction result of {}'.format(test_data))
        test_evaluator(sbert_model) 


def main(args):
    if args.is_test:
        test(args)
    else:
        # train(args)
        ngpus_per_node = args.ngpus 
        args.world_size = ngpus_per_node * 1
        mp.spawn(train,
            args=(ngpus_per_node, args),
            nprocs=ngpus_per_node,
            join=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BM training')
    parser.add_argument('--root-dir', type=str)
    parser.add_argument('--data-dir', type=str)
    parser.add_argument('--pt-model-dir', type=str)
    parser.add_argument('--output-model-dir', type=str)
    parser.add_argument('--larva-model', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--train-size', type=int)
    parser.add_argument('--valid-size', type=int)
    parser.add_argument('--data-name', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--warmup-rate', type=float)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--test-type', type=str)
    parser.add_argument('--is-test', default=False, action='store_true')
    parser.add_argument('--test-size', type=int)
    parser.add_argument('--ymd', type=str)
    parser.add_argument('--loss-type', type=str)
    parser.add_argument('--tensorboard-path', type=str)
    parser.add_argument('--train-dir', type=str)
    parser.add_argument('--ngpus', type=int)
    parser.add_argument('--start-gpu-num', type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--duplicates', default=1, type=int)
    parser.add_argument('--output-dim', type=int)

    # parser.add_argument('--rank', default=0, type=int)
    # parser.add_argument('--world-size', default=1, type=int)
    # parser.add_argument('--ngpus', '--ngpus-per-node-size', default=2, type=int)
    
    args = parser.parse_args()
    main(args)

