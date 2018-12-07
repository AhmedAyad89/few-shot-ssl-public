from run_exp import *
from run_exp import _get_model
import pickle
from fewshot.models.kmeans_utils import *
from tensorflow.contrib.tensorboard.plugins import projector
import csv



def get_data_embeddings(model, sess, dataset, num_examples, dim, pkl=None, tsv=None):
	num_classes = dataset.num_classes
	print(num_classes)
	features = np.zeros([num_classes, num_examples, dim])
	for i in range(num_classes):
		data = dataset.get_train_data(i)
		data = np.expand_dims(np.asarray(data, dtype=np.float64), 0)
		if np.shape(data)[1] == 0:
			print("EMPTY CLASS\n\n")
			continue
		feed = {model.training_data: data}
		embd = sess.run(model.embeddings, feed_dict=feed)
		features[i] = embd

	if pkl is not None:
		with open(FLAGS.pretrain + '/' + pkl, "wb") as fp:
			pickle.dump(features, fp)
	if tsv is not None:
		with open(FLAGS.pretrain + '/' + tsv, "w") as fp:
			writer = csv.writer(fp, delimiter='\t')
			for clss in features:
				for s in clss:
					writer.writerow(s)
		with open(FLAGS.pretrain + '/' + 'labels.tsv', "w") as fp:
			writer = csv.writer(fp)
			# writer.writerow('Label')
			for i in range(num_classes):
				for j in range(np.shape(features[i])[0] ):
					writer.writerow([str(i)])
	return features

def emperical_diag_cov(clusters, embeddings):
	s = np.shape(clusters)
	covar = np.zeros(s)
	for c, i in enumerate(clusters):
		embd = embeddings[i]
		d = embd - c
		covar[i] = np.mean(d * d , 0)
	return covar

def diagnose_clusters(train_embeddings, test_embeddings):
	train_protos = np.mean(train_embeddings, 1)
	test_protos = np.mean(test_embeddings, 1)

	train_emp_cov = emperical_diag_cov(train_protos, train_embeddings)
	test_emp_cov = emperical_diag_cov(test_protos, test_embeddings)

	train_protos_mean = np.mean(train_protos, 0)
	train_portos_cov =np.mean( np.square(train_protos - train_protos_mean), 0 )
	
	test_protos_mean = np.mean(test_protos, 0)
	test_portos_cov =np.mean( np.square(test_protos - test_protos_mean), 0 )

	probs_train = compute_logits(train_protos, train_embeddings)
	probs_test = compute_logits(test_protos, test_embeddings)
	probs_all = compute_logits(np.concatenate([train_protos, test_protos]), np.concatenate([train_embeddings, test_embeddings]))

	#compute portion of mass still on right class

def write_projection_embeddings(embeddings, meta_data, sess, log_dir):
	LOG_DIR = log_dir
	embedding_var = tf.Variable(embeddings, name='Embedding_of_fc6')
	sess.run(embedding_var.initializer)
	summary_writer = tf.summary.FileWriter(LOG_DIR)
	config = projector.ProjectorConfig()
	embedding = config.embeddings.add()
	embedding.tensor_name = embedding_var.name

	# Comment out if you don't have metadata
	embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

	# Comment out if you don't want sprites
	# embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite.png')
	# embedding.sprite.single_image_dim.extend([imgs.shape[1], imgs.shape[1]])

	projector.visualize_embeddings(summary_writer, config)
	saver = tf.train.Saver([embedding_var])
	saver.save(sess, os.path.join(LOG_DIR, 'model2.ckpt'), 1)


def main():
  if FLAGS.num_test == -1 and (FLAGS.dataset == "tiered-imagenet" or
                               FLAGS.dataset == 'mini-imagenet'):
    num_test = 5
  else:
    num_test = FLAGS.num_test
  config = get_config(FLAGS.dataset, FLAGS.model)
  nclasses_train = FLAGS.nclasses_train
  nclasses_eval = FLAGS.nclasses_eval

  # Which training split to use.
  train_split_name = 'train'
  if FLAGS.use_test:
    log.info('Using the test set')
    test_split_name = 'test'
  else:
    log.info('Not using the test set, using val')
    test_split_name = 'val'

  log.info('Use split `{}` for training'.format(train_split_name))
  # Whether doing 90 degree augmentation.
  if 'mini-imagenet' in FLAGS.dataset or 'tiered-imagenet' in FLAGS.dataset:
    _aug_90 = False
  else:
    _aug_90 = True

  nshot = FLAGS.nshot
  meta_train_dataset = get_dataset(
      FLAGS.dataset,
      train_split_name,
      nclasses_train,
      nshot,
      num_test=num_test,
      aug_90=_aug_90,
      num_unlabel=FLAGS.num_unlabel,
      shuffle_episode=False,
      seed=FLAGS.seed)
  meta_train_dataset = get_concurrent_iterator(
      meta_train_dataset, max_queue_size=100, num_threads=5)
  meta_test_dataset = get_dataset(
      FLAGS.dataset,
      test_split_name,
      nclasses_eval,
      nshot,
      num_test=num_test,
      aug_90=_aug_90,
      num_unlabel=FLAGS.num_unlabel,
      shuffle_episode=False,
      label_ratio=1,
      seed=FLAGS.seed)

  meta_test_dataset = get_concurrent_iterator(
      meta_test_dataset, max_queue_size=100, num_threads=5)
  m, mvalid = _get_model(config, nclasses_train, nclasses_eval)

  sconfig = tf.ConfigProto()
  sconfig.gpu_options.allow_growth = True
  with tf.Session(config=sconfig) as sess:
    ################################################
    list = [n for n in tf.all_variables() if "persistent_proto" not in n.name]
    list2 = [n for n in tf.all_variables() if "persistent_proto" in n.name]
    print(list)
    if FLAGS.pretrain is not None:
      ckpt = tf.train.latest_checkpoint(
          os.path.join(FLAGS.results, FLAGS.pretrain))
      saver = tf.train.Saver(var_list=list)
      saver.restore(sess, ckpt)
      if FLAGS.continue_train:
        train(sess, config, m, meta_train_dataset, mvalid, meta_test_dataset)
      if FLAGS.model == 'persistent':
        print('init persistent')
        meta_train_data = get_dataset(
          FLAGS.dataset,
          train_split_name,
          nclasses_train,
          nshot,
          num_test=num_test,
          aug_90=_aug_90,
          num_unlabel=FLAGS.num_unlabel,
          shuffle_episode=False,
          seed=FLAGS.seed)
        meta_test_data = get_dataset(
          FLAGS.dataset,
          test_split_name,
          nclasses_train,
          nshot,
          num_test=num_test,
          aug_90=_aug_90,
          num_unlabel=FLAGS.num_unlabel,
          shuffle_episode=False,
          seed=FLAGS.seed)
        _ = get_data_embeddings(mvalid, sess, meta_train_data, 20, 64, 'mini-imagenet-train-embeddings')
        _ = get_data_embeddings(mvalid, sess, meta_test_data, 20, 64, 'mini-imagenet-test-embeddings')

        num_train_classes = config.n_train_classes
        proto_dim = config.proto_dim
        protos = np.zeros([1, num_train_classes, proto_dim])
        for i in range(num_train_classes):
          data = meta_train_data.get_train_data(i)
          data = np.expand_dims(np.asarray(data, dtype=np.float64), 0)
          print(np.shape(data))
          if np.shape(data)[1] == 0:
            continue
          feed = {mvalid.training_data : data}
          proto = sess.run(mvalid.proto, feed_dict= feed)
          protos[0, i] = proto
        print(np.shape(protos), protos)
        mvalid._persistent_protos.load(protos)
        ################################################
    else:
      sess.run(tf.global_variables_initializer())
      train(sess, config, m, meta_train_dataset, mvalid, meta_test_dataset)

    results_train = evaluate(sess, mvalid, meta_train_dataset)
    results_test = evaluate(sess, mvalid, meta_test_dataset)

    log.info("Final train acc {:.3f}% ({:.3f}%)".format(
        results_train['acc'] * 100.0, results_train['acc_ci'] * 100.0))
    log.info("Final test acc {:.3f}% ({:.3f}%)".format(
        results_test['acc'] * 100.0, results_test['acc_ci'] * 100.0))


if __name__ == "__main__":
  main()