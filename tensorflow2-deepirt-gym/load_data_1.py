import numpy as np
from utils import getLogger

class DataLoader():
    def __init__(self, n_questions,n_skills, seq_len, separate_char):
        self.separate_char = separate_char
        self.n_questions = n_questions
        self.n_skills = n_skills
        self.seq_len = seq_len

    def load_data(self, path):
        s_data = []
        q_item_data = []
        qa_data = []
        with open(path, 'r') as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                # skip the number of sequence
                if line_idx%4 == 0:
                    continue
                # handle question_line
                elif line_idx%4 == 1:
                    q_item_tag_list = line.split(self.separate_char)
                    q_item_tag_list = [s for s in q_item_tag_list if s !='']
                # handle answer-line
                elif line_idx%4 == 2:
                    s_tag_list = line.split(self.separate_char)
                    #print(q_tag_list)
                    s_tag_list = [s for s in s_tag_list if s !='']
                    #a_tag_list = a_tag_list[a_tag_list!=""]
                    #print(q_tag_list)
                elif line_idx%4 == 3:
                    a_tag_list = line.split(self.separate_char)
                    a_tag_list = [s for s in a_tag_list if s !='']
                    # find the number of split for this sequence
                    n_split = len(q_item_tag_list) // self.seq_len
                    if len(q_item_tag_list) % self.seq_len != 0:
                        n_split += 1

                    for k in range(n_split):
                        # temporary container for each sequence
                        s_container = list()
                        q_item_container = list()
                        qa_container = list()

                        start_idx = k*self.seq_len
                        end_idx = min((k+1)*self.seq_len, len(a_tag_list))
                        #print(a_tag_list)
                        #print(start_idx,end_idx)
                        for i in range(start_idx, end_idx):
                            #print(i)
                            #print(q_tag_list[i])
                            s_value = int(s_tag_list[i])
                            q_item_value = int(q_item_tag_list[i])
                            a_value = int(a_tag_list[i])  # either be 0 or 1
                            qa_value = s_value + a_value * self.n_skills
                            s_container.append(s_value)
                            q_item_container.append(q_item_value)
                            qa_container.append(qa_value)
                        s_data.append(s_container)
                        q_item_data.append(q_item_container)
                        qa_data.append(qa_container)

        # convert it to numpy array
        s_data_array = np.zeros((len(s_data), self.seq_len))
        q_item_data_array = np.zeros((len(q_item_data), self.seq_len))
        qa_data_array = np.zeros((len(qa_data), self.seq_len))
        for i in range(len(q_item_data)):
            _s_data = s_data[i]
            _q_item_data = q_item_data[i]
            _qa_data = qa_data[i]
            s_data_array[i, :len(_s_data)] = _s_data
            q_item_data_array[i, :len(_q_item_data)] = _q_item_data
            qa_data_array[i, :len(_qa_data)] = _qa_data
        #項目IDとスキルIDがmainと逆なのでここで入れ替え
        return  q_item_data_array,s_data_array,qa_data_array


# import numpy as np
# from utils import getLogger




# class DataLoader():
#     def __init__(self, n_questions,n_skills, seq_len, separate_char):
#         self.separate_char = separate_char
#         self.n_questions = n_questions
#         self.n_skills = n_skills
#         self.seq_len = seq_len

#     def load_data(self, path):
#         s_data = []
#         q_item_data = []
#         qa_data = []
#         # with open(path, 'r') as f:
#         #     for line_idx, line in enumerate(f):
#         #         line = line.strip()
#         #         if line_idx == 18:
#         #           s_raw_tag_list = line.split(self.separate_char)
#         #           s_raw_tag_list = [s for s in s_raw_tag_list if s !='']
#         #           if mode == 'valid':
#         #             print(s_raw_tag_list)
#         with open(path, 'r') as f:
#             for line_idx, line in enumerate(f):
#                 line = line.strip()
#                 # skip the number of sequence
#                 if line_idx%4 == 0:
#                     continue
#                 # handle question_line
#                 elif line_idx%4 == 1:
#                     q_item_tag_list = line.split(self.separate_char)
#                     q_item_tag_list = [s for s in q_item_tag_list if s !='']
#                 # handle answer-line
#                 elif line_idx%4 == 2:
#                     s_tag_list = line.split(self.separate_char)
#                     #print(q_tag_list)
#                     s_tag_list = [s for s in s_tag_list if s !='']
#                     #a_tag_list = a_tag_list[a_tag_list!=""]
#                     #print(q_tag_list)
#                 elif line_idx%4 == 3:
#                     a_tag_list = line.split(self.separate_char)
#                     a_tag_list = [s for s in a_tag_list if s !='']
#                     # find the number of split for this sequence
#                     n_split = len(q_item_tag_list) // self.seq_len
#                     if len(q_item_tag_list) % self.seq_len != 0:
#                         n_split += 1

#                     for k in range(n_split):
#                         # temporary container for each sequence
#                         s_container = list()
#                         q_item_container = list()
#                         qa_container = list()

#                         start_idx = k*self.seq_len
#                         end_idx = min((k+1)*self.seq_len, len(a_tag_list))
#                         #print(a_tag_list)
#                         #print(start_idx,end_idx)
#                         for i in range(start_idx, end_idx):
#                             #print(i)
#                             #print(q_tag_list[i])
#                             # if mode == 'valid':
#                             #   s_true_value = int(s_raw_tag_list[i])
#                             # else:
#                             s_true_value = int(s_tag_list[i])
#                             s_value = int(s_tag_list[i])
#                             q_item_value = int(q_item_tag_list[i])
#                             a_value = int(a_tag_list[i])  # either be 0 or 1
#                             qa_value = s_true_value + a_value * self.n_skills

#                             s_container.append(s_value)
#                             q_item_container.append(q_item_value)
#                             qa_container.append(qa_value)
#                         s_data.append(s_container)
#                         q_item_data.append(q_item_container)
#                         qa_data.append(qa_container)

#         # convert it to numpy array
#         s_data_array = np.zeros((len(s_data), self.seq_len))
#         q_item_data_array = np.zeros((len(q_item_data), self.seq_len))
#         qa_data_array = np.zeros((len(qa_data), self.seq_len))
#         for i in range(len(q_item_data)):
#             _s_data = s_data[i]
#             _q_item_data = q_item_data[i]
#             _qa_data = qa_data[i]
#             s_data_array[i, :len(_s_data)] = _s_data
#             q_item_data_array[i, :len(_q_item_data)] = _q_item_data
#             qa_data_array[i, :len(_qa_data)] = _qa_data
#         #項目IDとスキルIDがmainと逆なのでここで入れ替え
#         return  q_item_data_array,s_data_array,qa_data_array
