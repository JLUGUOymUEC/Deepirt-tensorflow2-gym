import datetime
import os

class ModelConfigFactory():
    @staticmethod
    def create_model_config(args):
        if args.dataset == 'assist2009':
            return Assist2009Config(args).get_args()
        elif args.dataset == 'assist2009_akt':
            return Assist2009_akt_Config(args).get_args()
        elif args.dataset == 'assist2015_akt':
            return Assist2015_akt_Config(args).get_args()
        elif args.dataset == 'assist2015':
            return Assist2015Config(args).get_args()
        elif args.dataset == 'assist2017':
            return Assist2017Config(args).get_args()
        elif args.dataset == 'assist2017_akt':
            return Assist2017_akt_Config(args).get_args()
        elif args.dataset == 'statics2011':
            return Statics_Config(args).get_args()
        elif args.dataset == 'synthetic_irt':
            return SyntheticIRTConfig(args).get_args()
        elif args.dataset == 'synthetic_tirt':
            return SyntheticTIRTConfig(args).get_args()
        elif args.dataset == 'fsai':
            return FSAIConfig(args).get_args()
        elif args.dataset == 'KDD_item':
            return KDD_itemConfig(args).get_args()
        elif args.dataset == 'KDD_skill':
            return KDD_skillConfig(args).get_args()
        elif args.dataset == 'KDD_skill_item':
            return KDD_skill_itemConfig(args).get_args()
        elif args.dataset == 'statics_item':
            return Statics_itemConfig(args).get_args()
        elif args.dataset == 'statics_skill':
            return Statics_skillConfig(args).get_args()
        elif args.dataset == 'statics_skill_item':
            return Statics_skillitemConfig(args).get_args()
        elif args.dataset == 'assist2009_skill':
            return Ass_20000skillConfig(args).get_args()
        elif args.dataset == 'assist2009_item':
            return Ass_20000itemConfig(args).get_args()
        elif args.dataset == 'assist2009_skill_item':
            return Ass_20000skillitemConfig(args).get_args()
        elif args.dataset == '18item_skill_item':
            return item1_skillConfig(args).get_args()
        elif args.dataset == '7item_skill_item':
            return item2_skillConfig(args).get_args()
        elif args.dataset == 'risan_skill_item':
            return risan_skillConfig(args).get_args()
        elif args.dataset == 'simu':
            return Datasimu_Config(args).get_args()
        else:
            raise ValueError("The '{}' is not available".format(args.dataset))


class ModelConfig():
    def __init__(self, args):
        self.default_setting = self.get_default_setting()
        self.init_time = datetime.datetime.now().strftime("%Y-%m-%dT%H%M")

        self.args = args
        self.args_dict = vars(self.args)
        for arg in self.args_dict.keys():
            self._set_attribute_value(arg, self.args_dict[arg])

        self.set_result_log_dir()
        self.set_checkpoint_dir()
        self.set_tensorboard_dir()

    def get_args(self):
        return self.args

    def get_default_setting(self):
        default_setting = {}
        return default_setting

    def _set_attribute_value(self, arg, arg_value):
        self.args_dict[arg] = arg_value \
            if arg_value is not None \
            else self.default_setting.get(arg)

    def _get_model_config_str(self):
        model_config = 'b' + str(self.args.batch_size) \
                    + '_m' + str(self.args.memory_size) \
                    + '_q' + str(self.args.key_memory_state_dim) \
                    + '_qa' + str(self.args.value_memory_state_dim) \
                    + '_f' + str(self.args.summary_vector_output_dim)
        return model_config

    def set_result_log_dir(self):
        result_log_dir = os.path.join(
            './results',
            self.args.dataset,
            self._get_model_config_str(),
            self.init_time
        )
        self._set_attribute_value('result_log_dir', result_log_dir)

    def set_checkpoint_dir(self):
        checkpoint_dir = os.path.join(
            './models',
            self.args.dataset,
            self._get_model_config_str(),
            self.init_time
        )
        self._set_attribute_value('checkpoint_dir', checkpoint_dir)

    def set_tensorboard_dir(self):
        tensorboard_dir = os.path.join(
            './tensorboard',
            self.args.dataset,
            self._get_model_config_str(),
            self.init_time
        )
        self._set_attribute_value('tensorboard_dir', tensorboard_dir)

class Assist2009Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 110,
            'data_dir': './data/assist2009_updated',
            'data_name': 'assist2009_updated',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class Assist2009_akt_Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'mode':'both',
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 16891,
            'n_skills': 110,
            'data_dir': './data/assist2009_akt',
            'data_name': 'assist2009_pid',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class Datasimu_Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting

            'n_epochs': 30,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 50,
            'n_skills': 1,
            'data_dir': './data/simu',
            'data_name': 'simudata',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class Assist2017_akt_Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'mode':'both',
            'n_epochs': 30,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.005,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 3162,
            'n_skills': 102,
            'data_dir': './data/assist2017_akt',
            'data_name': 'assist2017_pid',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class Assist2015_akt_Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'mode':'both',
            'n_epochs': 30,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.005,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 3162,
            'n_skills': 100,
            'data_dir': './data/assist2015',
            'data_name': 'assist2015',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class Ass_20000skillConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 111,
            'data_dir': './data/assist_skill_20000',
            'data_name': 'assist_skill',
            # DKVMN param
            'memory_size': 20,
            'key_memory_state_dim': 20,
            'value_memory_state_dim': 20,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class Ass_20000itemConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 26662,
            'data_dir': './data/assistment_2009_20000_item',
            'data_name': 'assist_skill',
            # DKVMN param
            'memory_size': 20,
            'key_memory_state_dim': 20,
            'value_memory_state_dim': 20,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class Ass_20000skillitemConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.0005,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 26688,
            'n_skills': 111,
            'data_dir': './data/assist2009_skill_item',
            'data_name': 'assist_skill_item',
            # DKVMN param
            'memory_size': 20,
            'key_memory_state_dim': 20,
            'value_memory_state_dim': 40,
            'summary_vector_output_dim': 20,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class Assist2015Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'mode':'one',
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 100,
            'data_dir': './data/assist2015_akt',
            'data_name': 'assist2015',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class Assist2017Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'mode':'both',
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 3162,
            'n_skills': 102,
            'data_dir': './data/assist2017_akt',
            'data_name': 'assist2017_pid',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class Statics_Config(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'mode':'one',
            'n_epochs': 80,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 1223,
            'data_dir': './data/statics_akt',
            'data_name': 'statics',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class Statics_itemConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 1220,
            'data_dir': './data/sta_item',
            'data_name': 'new_statistics',
            # DKVMN param
            'memory_size': 20,
            'key_memory_state_dim': 20,
            'value_memory_state_dim': 20,
            'summary_vector_output_dim': 40,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class Statics_skillConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 41,
            'data_dir': './data/sta_skill',
            'data_name': 'new_statistics',
            # DKVMN param
            'memory_size': 10,
            'key_memory_state_dim': 10,
            'value_memory_state_dim': 5,
            'summary_vector_output_dim': 20,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class Statics_skillitemConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 10,
            'train': True,
            'show': True,
            'learning_rate': 0.002,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_skills': 98,#41,
            'n_questions' : 1223,#1223,
            'data_dir': './data/STATICS_skill_item',
            'data_name': 'statics_skill_item',
            # DKVMN param
            'memory_size': 20,
            'key_memory_state_dim': 20,
            'value_memory_state_dim': 40,
            'summary_vector_output_dim': 20,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class SyntheticIRTConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 50,
            'n_skills': 1,
            'data_dir': './data/synthetic/IRT_data',
            'data_name': 'item_skill',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting
class SyntheticTIRTConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 50,
            'n_skills': 1,
            'data_dir': './data/synthetic/TIRT_data',
            'data_name': 'item_skill',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class KDD_itemConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 550,
            'data_dir': './data/KDD_item',
            'data_name': 'assist_skill',
            # DKVMN param
            'memory_size': 20,
            'key_memory_state_dim': 20,
            'value_memory_state_dim': 40,
            'summary_vector_output_dim': 20,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting

class KDD_skillConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 73,
            'data_dir': './data/KDD_skill',
            'data_name': 'assist_skill',
            # DKVMN param
            'memory_size': 20,
            'key_memory_state_dim': 20,
            'value_memory_state_dim': 40,
            'summary_vector_output_dim': 20,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting
class KDD_skill_itemConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 30,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 200,
            'n_questions': 550,
            'n_skills': 73,
            'data_dir': './data/KDD_skill_item',
            'data_name': 'kdd_skill_item',
            # DKVMN param
            'memory_size': 20,
            'key_memory_state_dim': 20,
            'value_memory_state_dim': 40,
            'summary_vector_output_dim': 20,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting


class FSAIConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 32,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 50,
            'n_questions': 2266,
            'data_dir': './data/fsaif1tof3',
            'data_name': 'fsaif1tof3',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting
class item1_skillConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 5,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 18,
            'n_questions': 18,
            'n_skills': 1,
            'data_dir': './data/18item_skill_item',
            'data_name': '18item',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting
class item2_skillConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 5,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 7,
            'n_questions': 7,
            'n_skills': 1,
            'data_dir': './data/7item_skill_item',
            'data_name': '7item',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting
class risan_skillConfig(ModelConfig):
    def get_default_setting(self):
        default_setting = {
            # training setting
            'n_epochs': 50,
            'batch_size': 5,
            'train': True,
            'show': True,
            'learning_rate': 0.003,
            'max_grad_norm': 10.0,
            'use_ogive_model': False,
            # dataset param
            'seq_len': 125,
            'n_questions': 125,
            'n_skills': 1,
            'data_dir': './data/risan_skill_item',
            'data_name': 'risan',
            # DKVMN param
            'memory_size': 50,
            'key_memory_state_dim': 50,
            'value_memory_state_dim': 100,
            'summary_vector_output_dim': 50,
            # parameter for the SA Network and KCD network
            'student_ability_layer_structure': None,
            'question_difficulty_layer_structure': None,
            'discimination_power_layer_structure': None
        }
        return default_setting
