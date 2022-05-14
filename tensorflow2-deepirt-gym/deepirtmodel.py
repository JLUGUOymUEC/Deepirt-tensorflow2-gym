
# import  tensorflow as tf
# import tensorflow.keras as k
# import numpy as np
# from keras import regularizers


# class deepirt_dropout(tf.keras.Model):
#     def __init__(self, value_memory_state_dim, key_memory_state_dim,  memory_size, n_questions, seq_len,
#                   summary_vector_output_dim,n_skills,batch_size, reuse_flag):
#         super(deepirt_dropout, self).__init__()

#         self.value_memory_state_dim = value_memory_state_dim
#         self.key_memory_state_dim = key_memory_state_dim
#         self.memory_size = memory_size
#         self.n_questions = n_questions
#         self.n_skills = n_skills
#         self.batch_size = batch_size
#         self.seq_len = seq_len
#         self.key_memory_matrix = tf.Variable(tf.compat.v1.truncated_normal(shape=[self.memory_size, self.key_memory_state_dim],stddev=0.1,name= 'key_memory_matrix'))
#         self.value_memory_matrix = tf.Variable(tf.compat.v1.truncated_normal(shape=[self.memory_size, self.value_memory_state_dim],stddev=0.1,name='value_memory_matrix'))
#         self.value_memory_matrix = tf.tile(  # tile the number of value-memory by the number of batch
#             tf.expand_dims(self.value_memory_matrix, 0),  # make the batch-axis
#             tf.stack([batch_size, 1, 1])  
#         )
#         self.q_embed = tf.keras.layers.Embedding(self.n_questions + 1, self.key_memory_state_dim , name = 'question')
#         self.qa_embed = tf.keras.layers.Embedding(2*self.n_questions + 1, self.value_memory_state_dim , name = 'answer')
#         self.s_embed = tf.keras.layers.Embedding(self.n_skills + 1, self.key_memory_state_dim , name = 'skill')
#         # self.correlation_weight_operation = tf.keras.layers.Dense(name='Memory_correlation_weight', units=key_memory_state_dim,
#         #                                                     activation=tf.sigmoid)
#         #mogrifier operation
#         self.embedded_content_vector_m_round0 = tf.keras.layers.Dense(name='EmbeddedContentVectormRound0', units=self.batch_size, activation=tf.sigmoid)

#         self.value_memory_matrix_m_round1 = tf.keras.layers.Dense(name='ValueMatrixRound1', units=self.value_memory_state_dim, activation=tf.sigmoid)

#         self.embedded_content_vector_m_round2 = tf.keras.layers.Dense(name='EmbeddedContentVectormRound2', units=self.batch_size, activation=tf.sigmoid)
        
#         #DKVMN
#         self.erase_signal_operation = tf.keras.layers.Dense(name='erase_operation', units=value_memory_state_dim, activation=None)

#         self.erase_signal_operation_Mv = tf.keras.layers.Dense(name='erase_operation_mv', units=value_memory_state_dim, activation=None)

#         self.add_signal_operation = tf.keras.layers.Dense(name='add_operation', units=value_memory_state_dim, activation=None)
        
#         self.add_signal_operation_Mv = tf.keras.layers.Dense(name='add_operation_mv', units=value_memory_state_dim, activation=None)

#         self.zt_signal_operation = tf.keras.layers.Dense(name='zt_operation', units=value_memory_state_dim, activation=None)

#         self.zt_signal_operation_Mv = tf.keras.layers.Dense(name='zt_operation_mv', units=value_memory_state_dim, activation=None)

#         self.student_ability_1_operation = tf.keras.layers.Dense(name='StudentAbilityOutputLayer1', units=self.memory_size, activation=None)

#         self.question_difficulty_1_operation = tf.keras.layers.Dense(name='QuestionDifficultyOutputLayer1', units=summary_vector_output_dim, activation=tf.nn.tanh)

#         self.student_ability_operation , self.skill_difficulty_operation , self.question_difficulty_operation = list(), list(), list()
#         #kernel_regularizer=regularizers.l2(0.08),lambda x : tf.nn.leaky_relu(x, alpha=0.01)
#         for i in range(10):
#           student_ability_operation =  tf.keras.layers.Dense(name='StudentAbilityOutputLayer'+str(i+1), units=summary_vector_output_dim,kernel_regularizer=regularizers.l2(0.05),  activation=None)
#           self.student_ability_operation.append(student_ability_operation)

#         for i in range(10):
#           skill_difficulty_operation =  tf.keras.layers.Dense(name='QuestionskillDifficultyOutputLayer'+str(i+1), units=summary_vector_output_dim,kernel_regularizer=regularizers.l2(0.05), activation=None)
#           self.skill_difficulty_operation.append(skill_difficulty_operation)

#         for i in range(10):
#           question_difficulty_operation =  tf.keras.layers.Dense(name='QuestionDifficultyOutputLayer'+str(i+1), units=summary_vector_output_dim,kernel_regularizer=regularizers.l2(0.05),  activation=None)
#           self.question_difficulty_operation.append(question_difficulty_operation)

#         self.AlphaDropout = tf.keras.layers.Dropout(rate = 0.5)
#         self.BatchNormalization = tf.keras.layers.BatchNormalization(momentum=0.90,)

        
#         self.question_difficulty_operation_final = tf.keras.layers.Dense(name='QuestionDifficultyOutputLayerFinal', units=1, activation=None)
        
#         self.skill_difficulty_1_operation = tf.keras.layers.Dense(name='QuestionskillDifficultyOutputLayer1', units=summary_vector_output_dim, activation=tf.nn.tanh)

#         self.skill_difficulty_operation_final = tf.keras.layers.Dense(name='QuestionskillDifficultyOutputLayerFinal', units=1, activation=None)
        
#         # self.summary_vector_operation = tf.keras.layers.Dense(name='summary_operation', units=summary_vector_output_dim , activation=tf.nn.tanh)

#         # self.student_ablity_operation = tf.keras.layers.Dense(name='StudentAbilityOutputLayer', units=1, activation=None )
        
#         # self.question_difficult_operation = tf.keras.layers.Dense(name='QusetionDifficultOutputLayer', units=1, activation=tf.nn.tanh)

#     def call(self, input, training):
#         q_data, qa_data, s_data = input

#         pred_z_values = list()
#         student_abilities = list()
#         question_difficulties = list()

#         q_embed_data = self.q_embed(q_data)
#         qa_embed_data = self.qa_embed(qa_data)
#         s_embed_data = self.s_embed(s_data)
#         sliced_q_embed_data = tf.split(
#             value=q_embed_data, num_or_size_splits=self.seq_len, axis=1
#         )
#         sliced_qa_embed_data = tf.split(
#             value=qa_embed_data, num_or_size_splits=self.seq_len, axis=1
#         )
#         sliced_s_embed_data = tf.split(
#             value=s_embed_data, num_or_size_splits=self.seq_len, axis=1
#         )
#         pred_z_values = list()
#         student_abilities = list()
#         question_difficulties = list()

#         for i in range(self.seq_len):
#           # valuememorymatrix 50 100 correlationweight 100
          
#           embedded_query_vector = tf.squeeze(sliced_q_embed_data[i], 1)#50
#           embedded_content_vector = tf.squeeze(sliced_qa_embed_data[i], 1)#100
#           embedded_skill_vector = tf.squeeze(sliced_s_embed_data[i], 1)
#           embedding_result = tf.matmul(embedded_query_vector, tf.transpose(self.key_memory_matrix))
#           correlation_weight = tf.nn.softmax(embedding_result)
#           value_memory_matrix_reshaped = tf.reshape(self.value_memory_matrix, [-1, self.value_memory_state_dim])
#           correlation_weight_reshaped = tf.reshape(correlation_weight, [-1, 1])

#           _read_result = tf.multiply(value_memory_matrix_reshaped, correlation_weight_reshaped, name = 'corr')  # row-wise multiplication
#           read_result = tf.reshape(_read_result, [-1, self.memory_size, self.value_memory_state_dim])
#           read_content = tf.reduce_sum(read_result, axis=1, keepdims=False)
          
#           # #mogrifier
#           # value_memory_matrix_m_0 = self.value_memory_matrix
#           # embedded_content_vector_m_0 = embedded_content_vector * tf.transpose(self.embedded_content_vector_m_round0(tf.transpose(tf.reshape(value_memory_matrix_m_0, [-1, self.value_memory_state_dim]))))
#           # value_memory_matrix_m_1 = value_memory_matrix_m_0 * tf.reshape(self.value_memory_matrix_m_round1(embedded_content_vector),[-1, 1, self.value_memory_state_dim])
#           # embedded_content_vector_m_2 = embedded_content_vector_m_0 * tf.transpose(self.embedded_content_vector_m_round2(tf.transpose(tf.reshape(value_memory_matrix_m_1, [-1, self.value_memory_state_dim]))))

#           # embedded_content_vector = embedded_content_vector_m_2
#           # self.value_memory_matrix = value_memory_matrix_m_1

#           # value_memory_matrix_reshaped_hpre = tf.reshape(self.value_memory_matrix, [self.batch_size, -1])

#           # erase_signal = self.erase_signal_operation(embedded_content_vector)
#           # erase_signal_mv = self.erase_signal_operation_Mv(value_memory_matrix_reshaped_hpre)

#           # erase_signal = tf.sigmoid(erase_signal  +  erase_signal_mv)

#           # zt_signal = self.zt_signal_operation(embedded_content_vector)
#           # zt_signal_mv = self.zt_signal_operation_Mv(value_memory_matrix_reshaped_hpre)
          
#           # zt_signal = tf.sigmoid(zt_signal  +  zt_signal_mv)

#           # add_signal = self.add_signal_operation(zt_signal)
#           # add_signal_mv = self.add_signal_operation_Mv(value_memory_matrix_reshaped_hpre)

#           # add_signal = tf.nn.tanh(add_signal  +  add_signal_mv)

#           ##rawdeepirt
#           erase_signal = self.erase_signal_operation(embedded_content_vector)

#           add_signal = self.add_signal_operation(embedded_content_vector)


#           # # reshape from (batch_size, value_memory_state_dim) to (batch_size, 1, value_memory_state_dim)
#           erase_reshaped = tf.reshape(erase_signal, [-1, 1, self.value_memory_state_dim])
#           # reshape from (batch_size, value_memory_state_dim) to (batch_size, 1, value_memory_state_dim)
#           add_reshaped = tf.reshape(add_signal, [-1, 1, self.value_memory_state_dim])
#           # reshape from (batch_size, memory_size) to (batch_size, memory_size, 1)
#           cw_reshaped = tf.reshape(correlation_weight, [-1, self.memory_size, 1])

#           # # erase_mul/add_mul: Shape (batch_size, memory_size, value_memory_state_dim)
#           erase_mul = tf.multiply(erase_reshaped, cw_reshaped, name= 'erase')
#           add_mul = tf.multiply(add_reshaped, cw_reshaped, name = 'add')

#           # Update value memory
#           new_value_memory_matrix = self.value_memory_matrix * (1 - erase_mul)  # erase memory
#           new_value_memory_matrix += add_mul
#           self.value_memory_matrix = new_value_memory_matrix

#           # mastery_level_prior_difficulty = tf.concat([read_content, embedded_query_vector], 1)

#           student_ability_1 = self.student_ability_1_operation(read_content)
          
#           for i in range(0):
#             student_ability_1 = self.student_ability_operation[i](student_ability_1)
#             # student_ability_1 = self.BatchNormalization(student_ability_1, training = False)
#             student_ability_1 = tf.nn.selu(student_ability_1)
            
#             # student_ability_1 = self.AlphaDropout(student_ability_1)


#           student_ability = tf.reduce_sum(tf.multiply(correlation_weight,student_ability_1),axis=1)

#           student_ability= tf.reshape(student_ability,[self.batch_size,1])

#           question_difficulty_1 = self.question_difficulty_1_operation(embedded_query_vector)

#           for i in range(0):
            
#             question_difficulty_1 = self.question_difficulty_operation[i](question_difficulty_1)
#             # question_difficulty_1 = self.BatchNormalization(question_difficulty_1, training = False)
#             question_difficulty_1 = tf.nn.selu(question_difficulty_1)
            
#             # question_difficulty_1 = self.AlphaDropout(question_difficulty_1)

#           question_difficulty = self.question_difficulty_operation_final(question_difficulty_1)

#           skill_difficulty_1 = self.skill_difficulty_1_operation(embedded_skill_vector)

#           for i in range(0):
#             skill_difficulty_1 = self.skill_difficulty_operation[i](skill_difficulty_1)
#             # skill_difficulty_1 = self.BatchNormalization(skill_difficulty_1, training = False)
#             skill_difficulty_1 = tf.nn.selu(skill_difficulty_1)
            
#             # skill_difficulty_1 = self.AlphaDropout(skill_difficulty_1)

#           skill_difficulty = self.skill_difficulty_operation_final(skill_difficulty_1)

#           # summary_vector = self.summary_vector_operation(mastery_level_prior_difficulty)

#           # student_ability = self.student_ablity_operation(summary_vector)

#           # question_difficulty = self.question_difficult_operation(embedded_query_vector)

#           pred_z_value = 3.0*student_ability - question_difficulty - skill_difficulty


#           pred_z_values.append(pred_z_value)
#           student_abilities.append(student_ability)
#           question_difficulties.append(question_difficulty)

#         pred_z_values = tf.reshape(
#             tf.stack(pred_z_values, axis=1), 
#             [self.batch_size , self.seq_len]
#         )
#         # student_abilities = tf.reshape(
#         #     tf.stack(student_abilities, axis=1),
#         #     [32, self.seq_len]
#         # )
#         # question_difficulties = tf.reshape(
#         #     tf.stack(question_difficulties, axis=1),
#         #     [32, self.seq_len]
#         # )
#         return pred_z_values




import  tensorflow as tf
import tensorflow.keras as k
import numpy as np


class deepirt_dropout(tf.keras.Model):
    def __init__(self, value_memory_state_dim, key_memory_state_dim,  memory_size, n_questions, seq_len,
                  summary_vector_output_dim,n_skills,batch_size, reuse_flag):
        super(deepirt_dropout, self).__init__()

        self.value_memory_state_dim = value_memory_state_dim
        self.key_memory_state_dim = key_memory_state_dim
        self.memory_size = memory_size
        self.n_questions = n_questions
        self.n_skills = n_skills
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.key_memory_matrix = tf.Variable(tf.compat.v1.truncated_normal(shape=[self.memory_size, self.key_memory_state_dim],stddev=0.1,name= 'key_memory_matrix'))
        self.value_memory_matrix = tf.Variable(tf.compat.v1.truncated_normal(shape=[self.memory_size, self.value_memory_state_dim],stddev=0.1,name='value_memory_matrix'))
        self.value_memory_matrix = tf.tile(  # tile the number of value-memory by the number of batch
            tf.expand_dims(self.value_memory_matrix, 0),  # make the batch-axis
            tf.stack([batch_size, 1, 1])  
        )
        self.q_embed = tf.keras.layers.Embedding(self.n_skills + 1, self.key_memory_state_dim , name = 'question')
        self.qa_embed = tf.keras.layers.Embedding(2*self.n_skills + 1, self.value_memory_state_dim , name = 'answer')
        self.s_embed = tf.keras.layers.Embedding(self.n_questions + 1, self.key_memory_state_dim , name = 'skill')
        # self.correlation_weight_operation = tf.keras.layers.Dense(name='Memory_correlation_weight', units=key_memory_state_dim,
        #                                                     activation=tf.sigmoid)
        #mogrifier operation

        self.embedded_content_vector_m_round0 = tf.keras.layers.Dense(name='EmbeddedContentVectormRound0', units=self.batch_size, activation=tf.sigmoid)

        self.value_memory_matrix_m_round1 = tf.keras.layers.Dense(name='ValueMatrixRound1', units=self.value_memory_state_dim, activation=tf.sigmoid)

        self.embedded_content_vector_m_round2 = tf.keras.layers.Dense(name='EmbeddedContentVectormRound2', units=self.batch_size, activation=tf.sigmoid)
        
        self.value_memory_matrix_m_round3 = tf.keras.layers.Dense(name='ValueMatrixRound3', units=self.value_memory_state_dim, activation=tf.sigmoid)

        self.embedded_content_vector_m_round4 = tf.keras.layers.Dense(name='EmbeddedContentVectormRound4', units=self.batch_size, activation=tf.sigmoid)          
        
        self.value_memory_matrix_m_round5 = tf.keras.layers.Dense(name='ValueMatrixRound5', units=self.value_memory_state_dim, activation=tf.sigmoid)

        self.embedded_content_vector_m_round6 = tf.keras.layers.Dense(name='EmbeddedContentVectormRound6', units=self.batch_size, activation=tf.sigmoid)          
        
        #DKVMN
        
        self.erase_signal_operation = tf.keras.layers.Dense(name='erase_operation', units=value_memory_state_dim, activation=None)

        self.erase_signal_operation_Mv = tf.keras.layers.Dense(name='erase_operation_mv', units=value_memory_state_dim, activation=None)

        self.add_signal_operation = tf.keras.layers.Dense(name='add_operation', units=value_memory_state_dim, activation=None)
        
        self.add_signal_operation_Mv = tf.keras.layers.Dense(name='add_operation_mv', units=value_memory_state_dim, activation=None)

        self.zt_signal_operation = tf.keras.layers.Dense(name='zt_operation', units=value_memory_state_dim, activation=None)

        self.zt_signal_operation_Mv = tf.keras.layers.Dense(name='zt_operation_mv', units=value_memory_state_dim, activation=None)

        self.student_ability_1_operation = tf.keras.layers.Dense(name='StudentAbilityOutputLayer1', units=self.memory_size, activation=None)

        self.question_difficulty_1_operation = tf.keras.layers.Dense(name='QuestionDifficultyOutputLayer1', units=summary_vector_output_dim, activation=tf.nn.tanh)

        # self.student_ability_operation , self.skill_difficulty_operation , self.question_difficulty_opertion = list(), list(), list()
        
        # for i in range(5):
        #   student_ability_operation =  tf.keras.layers.Dense(name='StudentAbilityOutputLayer'+str(i+1), units=summary_vector_output_dim, activation=lambda x : tf.nn.leaky_relu(x, alpha=0.2))
        #   self.student_ability_operation.append(student_ability_operation)

        # for i in range(5):
        #   skill_difficulty_operation =  tf.keras.layers.Dense(name='QuestionskillDifficultyOutputLayer'+str(i+1), units=summary_vector_output_dim, activation=lambda x : tf.nn.leaky_relu(x, alpha=0.2))
        #   self.skill_difficulty_operation.append(skill_difficulty_operation)

        # for i in range(5):
        #   question_difficulty_opertion =  tf.keras.layers.Dense(name='QuestionDifficultyOutputLayer'+str(i+1), units=summary_vector_output_dim, activation=lambda x : tf.nn.leaky_relu(x, alpha=0.2))
        #   self.question_difficulty_opertion.append(question_difficulty_opertion)

        # self.AlphaDropout = tf.keras.layers.AlphaDropout(rate = 0.5)
        # self.BatchNormalization = tf.keras.layers.BatchNormalization()

        
        self.question_difficulty_operation = tf.keras.layers.Dense(name='QuestionDifficultyOutputLayer', units=1, activation=None)
        
        self.skill_difficulty_1_operation = tf.keras.layers.Dense(name='QuestionskillDifficultyOutputLayer1', units=summary_vector_output_dim, activation=tf.nn.tanh)

        self.skill_difficulty_operation = tf.keras.layers.Dense(name='QuestionskillDifficultyOutputLayer', units=1, activation=None)
        # self.summary_vector_operation = tf.keras.layers.Dense(name='summary_operation', units=summary_vector_output_dim , activation=tf.nn.tanh)

        # self.student_ablity_operation = tf.keras.layers.Dense(name='StudentAbilityOutputLayer', units=1, activation=None )
        
        # self.question_difficult_operation = tf.keras.layers.Dense(name='QusetionDifficultOutputLayer', units=1, activation=tf.nn.tanh)

    def call(self, input, training):
        q_data, qa_data, s_data = input

        pred_z_values = list()
        student_abilities = list()
        question_difficulties = list()

        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)
        s_embed_data = self.s_embed(s_data)
        sliced_q_embed_data = tf.split(
            value=q_embed_data, num_or_size_splits=self.seq_len, axis=1
        )
        sliced_qa_embed_data = tf.split(
            value=qa_embed_data, num_or_size_splits=self.seq_len, axis=1
        )
        sliced_s_embed_data = tf.split(
            value=s_embed_data, num_or_size_splits=self.seq_len, axis=1
        )
        pred_z_values = list()
        student_abilities = list()
        question_difficulties = list()

        for i in range(self.seq_len):
          # valuememorymatrix 50 100 correlationweight 100
          
          embedded_query_vector = tf.squeeze(sliced_q_embed_data[i], 1)#50
          embedded_content_vector = tf.squeeze(sliced_qa_embed_data[i], 1)#100
          embedded_skill_vector = tf.squeeze(sliced_s_embed_data[i], 1)
          embedding_result = tf.matmul(embedded_query_vector, tf.transpose(self.key_memory_matrix))
          correlation_weight = tf.nn.softmax(embedding_result)
          value_memory_matrix_reshaped = tf.reshape(self.value_memory_matrix, [-1, self.value_memory_state_dim])
          correlation_weight_reshaped = tf.reshape(correlation_weight, [-1, 1])

          _read_result = tf.multiply(value_memory_matrix_reshaped, correlation_weight_reshaped, name = 'corr')  # row-wise multiplication
          read_result = tf.reshape(_read_result, [-1, self.memory_size, self.value_memory_state_dim])
          read_content = tf.reduce_sum(read_result, axis=1, keepdims=False)
          
          # #mogrifier
          # value_memory_matrix_m_0 = self.value_memory_matrix
          # embedded_content_vector_m_0 = embedded_content_vector *1* tf.transpose(self.embedded_content_vector_m_round0(tf.transpose(tf.reshape(value_memory_matrix_m_0, [-1, self.value_memory_state_dim]))))
          # value_memory_matrix_m_1 = value_memory_matrix_m_0 * tf.reshape(self.value_memory_matrix_m_round1(embedded_content_vector_m_0),[-1, 1, self.value_memory_state_dim])
          # embedded_content_vector_m_1 = embedded_content_vector_m_0 *1* tf.transpose(self.embedded_content_vector_m_round2(tf.transpose(tf.reshape(value_memory_matrix_m_1, [-1, self.value_memory_state_dim]))))
          # value_memory_matrix_m_2 = value_memory_matrix_m_1 * tf.reshape(self.value_memory_matrix_m_round3(embedded_content_vector_m_0),[-1, 1, self.value_memory_state_dim])
          # embedded_content_vector_m_2 = embedded_content_vector_m_1 *1* tf.transpose(self.embedded_content_vector_m_round4(tf.transpose(tf.reshape(value_memory_matrix_m_2, [-1, self.value_memory_state_dim]))))
          # value_memory_matrix_m_3 = value_memory_matrix_m_1 * tf.reshape(self.value_memory_matrix_m_round5(embedded_content_vector_m_0),[-1, 1, self.value_memory_state_dim])
          # embedded_content_vector_m_3 = embedded_content_vector_m_2 *1* tf.transpose(self.embedded_content_vector_m_round6(tf.transpose(tf.reshape(value_memory_matrix_m_3, [-1, self.value_memory_state_dim]))))
          

          # value_memory_matrix_m_0 = self.value_memory_matrix
          # embedded_content_vector_m_0 = embedded_content_vector *1* self.embedded_content_vector_m_round0(tf.reshape(value_memory_matrix_m_0, [self.batch_size, -1]))
          # value_memory_matrix_m_1 = value_memory_matrix_m_0 *1* tf.reshape(self.value_memory_matrix_m_round1(embedded_content_vector),[-1, 1, self.value_memory_state_dim])
          # embedded_content_vector_m_2 = embedded_content_vector_m_0 *1* self.embedded_content_vector_m_round2(tf.reshape(value_memory_matrix_m_1, [self.batch_size, -1]))
          

          # embedded_content_vector = embedded_content_vector_m_3
          # self.value_memory_matrix = value_memory_matrix_m_3

          # value_memory_matrix_reshaped_hpre = tf.reshape(self.value_memory_matrix, [self.batch_size, -1])

          # erase_signal = self.erase_signal_operation(embedded_content_vector)
          # erase_signal_mv = self.erase_signal_operation_Mv(value_memory_matrix_reshaped_hpre)

          # erase_signal = tf.sigmoid(erase_signal  +  erase_signal_mv)

          # zt_signal = self.zt_signal_operation(embedded_content_vector)
          # zt_signal_mv = self.zt_signal_operation_Mv(value_memory_matrix_reshaped_hpre)
          
          # zt_signal = tf.sigmoid(zt_signal  +  zt_signal_mv)

          # add_signal = self.add_signal_operation(zt_signal)
          # add_signal_mv = self.add_signal_operation_Mv(value_memory_matrix_reshaped_hpre)

          # add_signal = tf.nn.tanh(add_signal  +  add_signal_mv)

          ##rawdeepirt
          erase_signal = self.erase_signal_operation(embedded_content_vector)

          add_signal = self.add_signal_operation(embedded_content_vector)


          # # reshape from (batch_size, value_memory_state_dim) to (batch_size, 1, value_memory_state_dim)
          erase_reshaped = tf.reshape(erase_signal, [-1, 1, self.value_memory_state_dim])
          # reshape from (batch_size, value_memory_state_dim) to (batch_size, 1, value_memory_state_dim)
          add_reshaped = tf.reshape(add_signal, [-1, 1, self.value_memory_state_dim])
          # reshape from (batch_size, memory_size) to (batch_size, memory_size, 1)
          cw_reshaped = tf.reshape(correlation_weight, [-1, self.memory_size, 1])

          # # erase_mul/add_mul: Shape (batch_size, memory_size, value_memory_state_dim)
          erase_mul = tf.multiply(erase_reshaped, cw_reshaped, name= 'erase')
          add_mul = tf.multiply(add_reshaped, cw_reshaped, name = 'add')

          # Update value memory
          new_value_memory_matrix = self.value_memory_matrix * (1 - erase_mul)  # erase memory
          new_value_memory_matrix += add_mul
          self.value_memory_matrix = new_value_memory_matrix

          # mastery_level_prior_difficulty = tf.concat([read_content, embedded_query_vector], 1)

          student_ability_1 = self.student_ability_1_operation(read_content)

          student_ability = tf.reduce_sum(tf.multiply(correlation_weight,student_ability_1),axis=1)

          student_ability= tf.reshape(student_ability,[self.batch_size,1])

          question_difficulty_1 = self.question_difficulty_1_operation(embedded_query_vector)

          question_difficulty = self.question_difficulty_operation(question_difficulty_1)

          skill_difficulty_1 = self.skill_difficulty_1_operation(embedded_skill_vector)

          skill_difficulty = self.skill_difficulty_operation(skill_difficulty_1)

          pred_z_value = 3.0*student_ability - question_difficulty - skill_difficulty


          pred_z_values.append(pred_z_value)
          student_abilities.append(student_ability)
          question_difficulties.append(question_difficulty)

        pred_z_values = tf.reshape(
            tf.stack(pred_z_values, axis=1), 
            [self.batch_size , self.seq_len]
        )
        # student_abilities = tf.reshape(
        #     tf.stack(student_abilities, axis=1),
        #     [32, self.seq_len]
        # )
        # question_difficulties = tf.reshape(
        #     tf.stack(question_difficulties, axis=1),
        #     [32, self.seq_len]
        # )
        return tf.sigmoid(pred_z_values)


