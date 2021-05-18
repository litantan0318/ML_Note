import os
import sys
import tensorflow as tf
import numpy as np
import configparser

import inputs


class ModelManager:
    def __init__(self, name, params=None):
        self.default_params = {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "batch_size": 128,
        }
        self.model_parms = {}
        self.name = name
        self.params = params
        self.gen_params = self.gen_params()

    def _gen_model(self, build_model_fn, feature_cols_fn, default_feature_cols_fn):
        if feature_cols_fn is None:
            feature_cols_fn = default_feature_cols_fn
        self.model_parms = self.params
        model_fn = build_model_fn(self.params, feature_cols_fn)
        return model_fn
    
    def build_model(self, feature_cols_fn=None):
        model_fn = None
        if self.name == "fm":
            from notes.FM.model import build_model_fn, default_feature_cols_fn
            model_fn = self._gen_model(build_model_fn, feature_cols_fn, default_feature_cols_fn)
        if self.name == "ffm":
            from notes.FFM.model import build_model_fn, default_feature_cols_fn
            model_fn = self._gen_model(build_model_fn, feature_cols_fn, default_feature_cols_fn)
        if self.name == "deep_fm":
            from notes.Deep_FM.model import build_model_fn, default_feature_cols_fn
            model_fn = self._gen_model(build_model_fn, feature_cols_fn, default_feature_cols_fn)
        if self.name == "dcn":
            from notes.DCN.model import build_model_fn, default_feature_cols_fn
            model_fn = self._gen_model(build_model_fn, feature_cols_fn, default_feature_cols_fn)
        if self.name == "pnn":
            from notes.PNN.model import build_model_fn, default_feature_cols_fn
            model_fn = self._gen_model(build_model_fn, feature_cols_fn, default_feature_cols_fn)
        if self.name == "nfm":
            from notes.NFM.model import build_model_fn, default_feature_cols_fn
            model_fn = self._gen_model(build_model_fn, feature_cols_fn, default_feature_cols_fn)
        if self.name == "afm":
            from notes.AFM.model import build_model_fn, default_feature_cols_fn
            model_fn = self._gen_model(build_model_fn, feature_cols_fn, default_feature_cols_fn)
        if self.name == "mlr":
            from notes.MLR.model import build_model_fn, default_feature_cols_fn
            model_fn = self._gen_model(build_model_fn, feature_cols_fn, default_feature_cols_fn)
        if self.name == "din":
            from notes.DIN.model import build_model_fn, default_feature_cols_fn
            model_fn = self._gen_model(build_model_fn, feature_cols_fn, default_feature_cols_fn)
        return model_fn

    def build_input_fn(self):
        if self.name == "fm":
            return inputs.test_fm_input_fn
        if self.name == "ffm":
            return inputs.test_ffm_input_fn
        if self.name == "deep_fm":
            return inputs.test_deep_fm_input_fn
        if self.name == "dcn":
            return inputs.test_dcn_input_fn
        if self.name == "pnn":
            return inputs.test_pnn_input_fn
        if self.name == "nfm":
            return inputs.test_nfm_input_fn
        if self.name == "afm":
            return inputs.test_afm_input_fn
        if self.name == "mlr":
            return inputs.test_mlr_input_fn
        if self.name == "din":
            return inputs.test_din_input_fn

    def gen_params(self):
        if self.name in ("fm", "ffm"):
            if self.params is None:
                self.params = self.default_params
                self.params["embedding_size"] = 64
        if self.name == "deep_fm":
            if self.params is None:
                self.params = self.default_params
                self.params["embedding_size"] = 64
                # noinspection PyTypeChecker
                self.params["layers"] = (64, 64, 64)
        if self.name == "dcn":
            if self.params is None:
                self.params = self.default_params
                self.params["embedding_size"] = 64
                self.params["deep_layers"] = (64, 64, 64)
                self.params["cross_layers"] = (64, 64, 64)
        if self.name == "pnn":
            if self.params is None:
                self.params = self.default_params
                self.params["embedding_size"] = 64
                self.params["linear_units"] = 32
                self.params["cross_units"] = 32
                self.params["product_type"] = "inner"
                self.params["mlp_layers"] = [32, 32]
        if self.name == "nfm":
            if self.params is None:
                self.params = self.default_params
                self.params["embedding_size"] = 64
                self.params["deep_layers"] = [32, 32]
        if self.name == "afm":
            if self.params is None:
                self.params = self.default_params
                self.params["embedding_size"] = 64
                self.params["head_num"] = 2
        if self.name == "mlr":
            if self.params is None:
                self.params = self.default_params
                self.params["embedding_size"] = 64
                self.params["lr_nums"] = 10
        if self.name == "din":
            if self.params is None:
                self.params = self.default_params
                self.params["embedding_size"] = 128
                self.params["learning_rate"] = 0.01
                # self.params["beta1"] = 0.99
                self.params["max_step"] = 10000
                self.params["use_mlp"] = False
                self.params["his_len"] = 2
        else:
            print("name not set, use default params")
            print(self.default_params)
            if self.params is None:
                self.params = self.default_params


manager = ModelManager("din")


def train(cf):
    model_dir = "./test_model"
    cmd = "rm -rf %s" % model_dir
    os.system(cmd)

    # ==========  构建Estimator  ========== #
    config = tf.estimator.RunConfig(
        log_step_count_steps=cf.getint('train', 'eval_iter'),
        save_summary_steps=cf.getint('train', 'eval_iter'),
        save_checkpoints_steps=cf.getint('train', 'save_para_every'),
        keep_checkpoint_max=2
    )

    print(manager.build_model())
    print("="*20)
    model_dir = "./test_model"
    estimator = tf.estimator.Estimator(
        model_fn=manager.build_model(),
        model_dir=model_dir,
        params=manager.model_parms,
        config=config)

    # from tensorflow.python import debug as tf_debug
    # train_hooks = [tf_debug.LocalCLIDebugHook(ui_type="readline")]
    train_hooks = []
    eval_hooks = []

    # ==========  执行任务  ========== #
    train_spec = tf.estimator.TrainSpec(
        input_fn=manager.build_input_fn(),
        max_steps=manager.params.get("max_step", 10000),
        hooks=train_hooks)

    eval_spec = tf.estimator.EvalSpec(
        input_fn=manager.build_input_fn(),
        start_delay_secs=1,
        throttle_secs=60,
        steps=1000,
    )
    print("start train and eval")
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    print("variable names:")
    for var_name in estimator.get_variable_names():
        print(var_name)


def predict(cf):

    # ==========  构建Estimator  ========== #
    config = tf.estimator.RunConfig(
        log_step_count_steps=cf.getint('train', 'eval_iter'),
        save_summary_steps=cf.getint('train', 'eval_iter'),
        save_checkpoints_steps=cf.getint('train', 'save_para_every'),
        keep_checkpoint_max=2
    )

    print(manager.build_model())
    print("=" * 20)
    model_dir = "./test_model"
    estimator = tf.estimator.Estimator(
        model_fn=manager.build_model(),
        model_dir=model_dir,
        params=manager.model_parms,
        config=config)

    result = estimator.predict(manager.build_input_fn())
    return result


if __name__ == "__main__":
    cf = configparser.ConfigParser()
    cf.read("run.conf")
    print("tensorflow version:")
    print(tf.__version__)

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    train(cf)

    # np.set_printoptions(threshold=np.inf)
    # np.set_printoptions(suppress=True)
    # np.set_printoptions(precision=5)
    # result = predict(cf)
    # count = 0
    # for i in result:
    #     if count > 5:
    #         break
    #     if i["match"] == 1:
    #         print(i)
    #         count += 1