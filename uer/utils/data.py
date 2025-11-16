import os
import random
import pickle
import torch
from multiprocessing import Pool
from uer.utils.constants import *
from uer.utils.tokenizers import *
from uer.utils.misc import count_lines
from uer.utils.seed import set_seed


def mask_seq(src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
    vocab = tokenizer.vocab

    for i in range(len(src) - 1, -1, -1):
        if src[i] != PAD_ID:
            break
    src_no_pad = src[:i + 1]
    tokens_index, src_no_pad = create_index(src_no_pad, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length)
    if len(src_no_pad) < len(src):
        src = src_no_pad + (len(src) - len(src_no_pad)) * [PAD_ID]
    else:
        src = src_no_pad

    random.shuffle(tokens_index)
    num_to_predict = max(1, int(round(len(src_no_pad) * 0.15))) 
    tgt_mlm = []
    for index_set in tokens_index:
        if len(tgt_mlm) >= num_to_predict:
            break
        if whole_word_masking:
            i = index_set[0]
            mask_len = index_set[1]
            if len(tgt_mlm) + mask_len > num_to_predict:
                continue

            for j in range(mask_len):
                token = src[i + j]
                tgt_mlm.append((i + j, token))
                prob = random.random()
                if prob < 0.8:
                    src[i + j] = vocab.get(MASK_TOKEN)
                elif prob < 0.9:
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i + j] = rdi
        elif span_masking:
            i = index_set[0]
            span_len = index_set[1]
            if len(tgt_mlm) + span_len > num_to_predict:
                continue

            for j in range(span_len):
                token = src[i + j]
                tgt_mlm.append((i + j, token))
            prob = random.random()
            if prob < 0.8:
                for j in range(span_len):
                    src[i + j] = vocab.get(MASK_TOKEN)
            elif prob < 0.9:
                for j in range(span_len):
                    while True:
                        rdi = random.randint(1, len(vocab) - 1)
                        if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                            break
                    src[i + j] = rdi
        else:
            i = index_set[0]
            token = src[i]
            tgt_mlm.append((i, token))
            prob = random.random()
            if prob < 0.8: 
                src[i] = vocab.get(MASK_TOKEN)
            elif prob < 0.9: 
                while True:
                    rdi = random.randint(1, len(vocab) - 1)
                    if rdi not in [vocab.get(CLS_TOKEN), vocab.get(SEP_TOKEN), vocab.get(MASK_TOKEN), PAD_ID]:
                        break
                src[i] = rdi
    tgt_mlm = sorted(tgt_mlm, key=lambda x: x[0])
    return src, tgt_mlm


def create_index(src, tokenizer, whole_word_masking, span_masking, span_geo_prob, span_max_length):
    tokens_index = []
    span_end_position = -1
    vocab = tokenizer.vocab
    if whole_word_masking: 
        src_wwm = []
        src_length = len(src)
        has_cls, has_sep = False, False
        if src[0] == vocab.get(CLS_TOKEN):
            src = src[1:]
            has_cls = True
        if src[-1] == vocab.get(SEP_TOKEN):
            src = src[:-1]
            has_sep = True
        sentence = "".join(tokenizer.convert_ids_to_tokens(src)).replace('[UNK]', '').replace('##', '')
        import jieba
        wordlist = jieba.cut(sentence)
        if has_cls:
            src_wwm += [vocab.get(CLS_TOKEN)]
        for word in wordlist:
            position = len(src_wwm)
            src_wwm += tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
            if len(src_wwm) < src_length:
                tokens_index.append([position, len(src_wwm)-position])
        if has_sep:
            src_wwm += [vocab.get(SEP_TOKEN)]
        if len(src_wwm) > src_length:
            src = src_wwm[:src_length]
        else:
            src = src_wwm
    else:
        for (i, token) in enumerate(src):
            if token == vocab.get(CLS_TOKEN) or token == vocab.get(SEP_TOKEN) or token == PAD_ID:
                continue
            if not span_masking:
                tokens_index.append([i])
            else:
                if i < span_end_position:
                    continue
                span_len = get_span_len(span_max_length, span_geo_prob)
                span_end_position = i + span_len
                if span_end_position > len(src):
                    span_len = len(src) - i
                tokens_index.append([i, span_len])
    return tokens_index, src


def get_span_len(max_span_len, p):
    geo_prob_cum = [0.0]
    geo_prob = 1.0
    for i in range(max_span_len + 1):
        if i == 0:
            continue
        if i == 1:
            geo_prob *= p
            geo_prob_cum.append(geo_prob_cum[-1] + geo_prob)
        else:
            geo_prob *= (1 - p)
            geo_prob_cum.append(geo_prob_cum[-1] + geo_prob)

    prob = geo_prob_cum[-1] * random.random()
    for i in range(len(geo_prob_cum) - 1):
        if prob >= geo_prob_cum[i] and prob < geo_prob_cum[i + 1]:
            current_span_len = i + 1
    return current_span_len


def merge_dataset(dataset_path, workers_num):
    # Merge datasets.
    dataset_writer = open(dataset_path, "wb")
    for i in range(workers_num):
        tmp_dataset_reader = open("/mnt/data/zgm/ET-BERT/datasets/temp/dataset-tmp-" + str(i) + ".pt", "rb")
        while True:
            tmp_data = tmp_dataset_reader.read(2^20) 
            if tmp_data:
                dataset_writer.write(tmp_data)
            else:
                break
        tmp_dataset_reader.close()
        os.remove("/mnt/data/zgm/ET-BERT/datasets/temp/dataset-tmp-" + str(i) + ".pt")
    dataset_writer.close()


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """ truncate sequence pair to specific length """
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break
        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        if random.random() < 0.5: 
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def record_flow_start(corpus_path):
    starts = []
    with open(corpus_path, mode="r", encoding="utf-8") as f:
        i = 0
        while True:
            line = f.readline()
            if not line:
                break
            if line[:2] == "||":
                starts.append(i)
            i+=1
    starts.append(i)
    return starts

class Dataset(object):
    def __init__(self, args, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.corpus_path = args.corpus_path
        self.dataset_path = args.dataset_path
        self.seq_length = args.seq_length
        self.seed = args.seed
        self.dynamic_masking = args.dynamic_masking
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length
        self.docs_buffer_size = args.docs_buffer_size
        self.dup_factor = args.dup_factor

    def build_and_save(self, workers_num, split_by_flow=False):
        """
        Build dataset from the given corpus.
        Start workers_num processes and each process deals with a part of data.
        """
        lines_num = count_lines(self.corpus_path)
        print("Starting %d workers for building datasets ... " % workers_num)
        assert (workers_num >= 1)
        if workers_num == 1:
            self.worker(0, 0, lines_num)
        else:
            pool = Pool(workers_num)
            if split_by_flow:
                starts = record_flow_start(self.corpus_path)
            current_index = 0
            for i in range(workers_num):
                if split_by_flow:
                    # start = starts[current_index]
                    # for j in range(len(starts))[current_index:]:
                    #     if starts[j]-starts[current_index]>perburst:
                    #         current_index = j
                    #         break
                    # if i==workers_num-1:
                    #     current_index = len(starts)-1
                    # end = starts[current_index]
                    start = starts[i*(len(starts)-1)//workers_num]
                    end = starts[(i+1)*(len(starts)-1)//workers_num]
                else:
                    start = i * lines_num // workers_num
                    end = (i + 1) * lines_num // workers_num
                pool.apply_async(func=self.worker, args=[i, start, end])
            pool.close()
            pool.join()

        # Merge datasets.
        merge_dataset(self.dataset_path, workers_num)
        

    def worker(self, proc_id, start, end):
        raise NotImplementedError()


class DataLoader(object):
    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=False):
        self.tokenizer = args.tokenizer
        self.batch_size = batch_size
        self.instances_buffer_size = args.instances_buffer_size
        self.proc_id = proc_id
        self.proc_num = proc_num
        self.shuffle = shuffle
        self.dataset_reader = open(dataset_path, "rb")
        self.read_count = 0
        self.start = 0
        self.end = 0
        self.buffer = []
        self.vocab = args.vocab
        self.whole_word_masking = args.whole_word_masking
        self.span_masking = args.span_masking
        self.span_geo_prob = args.span_geo_prob
        self.span_max_length = args.span_max_length

    def _fill_buf(self):
        try:
            self.buffer = []
            while True:
                instance = pickle.load(self.dataset_reader)
                self.read_count += 1
                if (self.read_count - 1) % self.proc_num == self.proc_id: 
                    self.buffer.append(instance)
                    if len(self.buffer) >= self.instances_buffer_size:
                        break
        except EOFError:
            # Reach file end.
            self.dataset_reader.seek(0)

        if self.shuffle:
            random.shuffle(self.buffer)
        self.start = 0
        self.end = len(self.buffer)

    def _empty(self):
        return self.start >= self.end

    def __del__(self):
        self.dataset_reader.close()

class BertFlowDataset(Dataset):
    """
    Construct dataset for MLM and MIX tasks from the given corpus.
    Each document consists of multiple paragraphs,
    Each paragraph consists of multiple sentences,
    and each sentence occupies a single line.
    Paragraphs in corpus must be separated by empty lines.
    Documents in corpus must be separated by empty lines.
    """

    def __init__(self, args, vocab, tokenizer):
        super(BertFlowDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building dataset ... " % proc_id)
        print(start,end)
        set_seed(self.seed)
        flow_buffer = []
        flow_proto = []
        docs_buffer = []
        document = []
        pos = 0
        dataset_writer = open("/mnt/data/zgm/ET-BERT/datasets/temp/dataset-tmp-" + str(proc_id) + ".pt", "wb")
        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            try:
                #with open(self.corpus_path[:-4]+"_extra.txt", mode="r", encoding="utf-8") as fe:
                while pos < start:
                    f.readline()
                    pos += 1
                while True:
                    line = f.readline()
                    if pos==start and line[:2] != "||":
                        print("not flow start...")
                    pos += 1
                    if pos > end:
                        if len(docs_buffer) >= 1:
                            flow_buffer.append(docs_buffer)
                            
                        if len(flow_buffer) > 0:
                            try:
                                instances = self.build_instances(flow_buffer,flow_proto) 
                            except Exception as e:
                                print("has error2...", len(flow_buffer), len(flow_proto), e)
                            for instance in instances:
                                pickle.dump(instance, dataset_writer)
                        break

                    if not line.strip(): 
                        if len(document) >= 1:
                            docs_buffer.append(document)
                        document = []
                        continue
                    if line[:2] == "||" or not line: 
                        if len(docs_buffer) >= 1:
                            flow_buffer.append(docs_buffer)
                        docs_buffer = []
                    
                        flow_buffer_size = 0
                        for d in flow_buffer:
                            flow_buffer_size += len(d)
                        if flow_buffer_size > self.docs_buffer_size or pos>=end: 
                            print("Worker %d is building instances ... " % proc_id, len(flow_buffer),len(flow_proto))
                            try:
                                instances = self.build_instances(flow_buffer,flow_proto)
                            except:
                                print("has error1...")
                            print("Worker {} has {} instances. ".format(proc_id,len(instances)))
                            # Save instances.
                            for instance in instances: 
                                pickle.dump(instance, dataset_writer)
                            flow_buffer = []
                            flow_proto = []
                        if pos>=end or not line:
                            break
                        line = line[2:]
                        if line[0]=="4":
                            if "bigram" in self.corpus_path:
                                if line[42:44] == "06":
                                    flow_proto.append(0)
                                elif line[42:44] == "11":
                                    flow_proto.append(1)
                                else:
                                    print("not tcp or udp, ",line[42:44])
                            else:
                                if line[22:24] == "06":
                                    flow_proto.append(0)
                                elif line[22:24] == "11":
                                    flow_proto.append(1)
                        else:
                            print("find Ipv6!!")

                    sentence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(line)) 
                    if len(sentence) > 0:
                        document.append(sentence)

            except:
                print("has error...")
        print("Worker %d finished... " % proc_id)
        dataset_writer.close()

    def build_instances(self, all_documents, flow_proto):
        assert len(all_documents) == len(flow_proto)
        instances = []
        for _ in range(self.dup_factor):
            for doc_index in range(len(all_documents)):
                # if doc_index%500==0:
                #     print(doc_index)
                instances.extend(self.create_ins_from_doc(all_documents, doc_index, flow_proto))
        return instances
    def create_ins_from_doc(self, all_documents,document_index, flow_proto):
        document = all_documents[document_index] 
        max_num_tokens = self.seq_length - 3
        target_seq_length = max_num_tokens
        if random.random() < self.short_seq_prob: 
            target_seq_length = random.randint(2, max_num_tokens)
        instances = []
        i=0
        while i < len(document): 
            rnd = random.random()
            if i==len(document)-1 or rnd < 0.4: 
                a_end = 1
                if len(document[i]) >= 2: 
                    a_end = random.randint(1, len(document[i]) - 1)

                tokens_a = [] #seg 1
                for j in range(a_end):
                    tokens_a.extend(document[i][j])

                tokens_b = [] ##seg 2
                for j in range(a_end, len(document[i])):
                    tokens_b.extend(document[i][j])
                
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                if random.random()<0.5:  #A1 A2
                    label = 0 #
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
                else: #A2 A1
                    label = 1
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
            elif rnd < 0.6: 
                tokens_a = []
                for j in range(len(document[i])):
                    tokens_a.extend(document[i][j])
                next_burst_max_length = target_seq_length - len(tokens_a)

                for _ in range(20):
                    random_document_index = random.randint(0, len(all_documents) - 1)
                    if random_document_index != document_index:
                        break

                random_document = all_documents[random_document_index] 
                random_start = random.randint(0, len(random_document) - 1)
                burst_ind_end = random_start+1 #len(random_document) 
                tokens_b = []
                for burst_ind in range(random_start, burst_ind_end): 
                    for j in range(len(random_document[burst_ind])): 
                        tokens_b.extend(random_document[burst_ind][j])
                        if len(tokens_b) >= next_burst_max_length:
                            break
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                if random_document_index == document_index:
                    if random_start>=i:
                        label = 3 
                    else:
                        label = 4
                else:
                    label = 2 # A1A2 Z1Z2
                src = []
                src.append(self.vocab.get(CLS_TOKEN))
                src.extend(tokens_a)
                src.append(self.vocab.get(SEP_TOKEN))
                seg_pos = [len(src)]
                src.extend(tokens_b)
                src.append(self.vocab.get(SEP_TOKEN))
                seg_pos.append(len(src))
            else:
                tokens_a = []
                for j in range(len(document[i])):
                    tokens_a.extend(document[i][j])
                i+=1
                tokens_b = []
                for j in range(len(document[i])):
                    tokens_b.extend(document[i][j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)
                if random.random()<0.5:  #A1A2 B1B2
                    label = 3 
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
                else: #B1B2 A1A2
                    label = 4
                    src = []
                    src.append(self.vocab.get(CLS_TOKEN))
                    src.extend(tokens_b)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos = [len(src)]
                    src.extend(tokens_a)
                    src.append(self.vocab.get(SEP_TOKEN))
                    seg_pos.append(len(src))
            while len(src) != self.seq_length:
                src.append(PAD_ID)
            if not self.dynamic_masking:
                src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                instance = (src, tgt_mlm, label, seg_pos)
            else:
                instance = (src, label, seg_pos)
            instance += (flow_proto[document_index],)
            instances.append(instance)
            i+=1 
    
        return instances

class BertFlowDataLoader(DataLoader):
    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            tgt_mlm = []
            is_next = []
            seg = []
            proto = []

            masked_words_num = 0

            for ins in instances:
                if len(ins) == 5: 
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[2])
                    seg.append([1] * ins[3][0] + [2] * (ins[3][1] - ins[3][0]) + [PAD_ID] * (len(ins[0]) - ins[3][1]))
                    proto.append(ins[4])
                else:
                    src_single, tgt_mlm_single = mask_seq(ins[0], self.tokenizer, self.whole_word_masking, self.span_masking, self.span_geo_prob, self.span_max_length)
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    is_next.append(ins[1])
                    seg.append([1] * ins[2][0] + [2] * (ins[2][1] - ins[2][0]) + [PAD_ID] * (len(ins[0]) - ins[2][1]))
                    proto.append(ins[4])
            if masked_words_num == 0:
                continue

            yield torch.LongTensor(src), \
                torch.LongTensor(tgt_mlm), \
                torch.LongTensor(is_next), \
                torch.LongTensor(seg), \
                torch.LongTensor(proto)


# ============================================================================
# Multi-Modal Dataset for Contrastive Pre-training
# ============================================================================

class MultiModalDataset(Dataset):
    """
    多模态流量数据集，用于对比学习预训练

    从JSON文件读取流特征，构建包含：
    - Raw packet tokens (bigram)
    - Temporal tokens (IAT)
    - Size+Direction tokens
    """

    def __init__(self, args, vocab, tokenizer):
        super(MultiModalDataset, self).__init__(args, vocab, tokenizer)
        self.json_path = args.corpus_path  # 复用corpus_path参数，指向JSON文件

    def worker(self, proc_id, start, end):
        """构建预训练数据集"""
        import json

        print(f"Worker {proc_id} is building dataset...")
        set_seed(self.seed)

        dataset_writer = open(f"/tmp/dataset-tmp-{proc_id}.pt", "wb")

        # 读取JSON文件
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        flows = data['flows']
        total_flows = len(flows)

        # 分配给当前worker的流
        flows_for_worker = flows[start:end]

        print(f"Worker {proc_id}: processing {len(flows_for_worker)} flows")

        for flow in flows_for_worker:
            instances = self._create_instances_from_flow(flow)
            for instance in instances:
                pickle.dump(instance, dataset_writer)

        dataset_writer.close()
        print(f"Worker {proc_id} finished.")

    def _create_instances_from_flow(self, flow):
        """从单个流创建训练实例"""
        raw_tokens_list = flow['raw_tokens']
        temporal_tokens_list = flow['temporal_tokens']
        size_tokens_list = flow['size_tokens']

        num_packets = len(raw_tokens_list)

        if num_packets < 1:
            return []

        # 构建完整的token序列
        all_tokens = []
        all_token_types = []
        all_positions = []

        global_pos = 0

        for pkt_idx in range(num_packets):
            # Raw packet tokens
            raw_tokens = raw_tokens_list[pkt_idx]
            for token in raw_tokens:
                # 将hex转为int
                token_id = int(token, 16)
                all_tokens.append(token_id)
                all_token_types.append(0)  # type=0: raw packet
                all_positions.append(global_pos)
                global_pos += 1

            # Temporal tokens
            temporal_tokens = temporal_tokens_list[pkt_idx]
            for token_id in temporal_tokens:
                all_tokens.append(token_id)
                all_token_types.append(1)  # type=1: temporal
                all_positions.append(global_pos)
                global_pos += 1

            # Size+Direction token
            size_token = size_tokens_list[pkt_idx]
            all_tokens.append(size_token)
            all_token_types.append(2)  # type=2: size
            all_positions.append(global_pos)
            global_pos += 1

        # 截断到seq_length（完整包截断）
        truncated_tokens, truncated_types, truncated_positions = \
            self._truncate_to_complete_packets(
                all_tokens, all_token_types, all_positions
            )

        # 添加[CLS]和padding
        src = [self.vocab.get(CLS_TOKEN)] + truncated_tokens
        token_types = [0] + truncated_types  # [CLS]的type设为0
        positions = [0] + [p+1 for p in truncated_positions]

        # Padding
        padding_len = self.seq_length - len(src)
        if padding_len > 0:
            src += [PAD_ID] * padding_len
            token_types += [0] * padding_len
            positions += [0] * padding_len

        src = src[:self.seq_length]
        token_types = token_types[:self.seq_length]
        positions = positions[:self.seq_length]

        # 应用Masking（用于MBM辅助任务）
        if not self.dynamic_masking:
            masked_src, masked_labels = self._apply_masking(src, token_types)
        else:
            masked_src = src
            masked_labels = [0] * len(src)  # 动态masking在训练时处理

        # 构造实例
        instance = (
            masked_src,         # 输入序列
            masked_labels,      # mask标签（用于MBM）
            token_types,        # token类型（0=raw, 1=temporal, 2=size）
            positions,          # 位置索引
        )

        return [instance]

    def _truncate_to_complete_packets(self, tokens, types, positions):
        """
        截断到完整包（方案B）

        确保不会截断一个包的中间，保持包的完整性
        """
        if len(tokens) <= self.seq_length - 1:  # -1 for [CLS]
            return tokens, types, positions

        # 找到包的边界（size token的位置）
        packet_boundaries = [i for i, t in enumerate(types) if t == 2]

        # 找到最后一个完整包的结束位置
        max_len = self.seq_length - 1
        last_complete_idx = 0

        for boundary in packet_boundaries:
            if boundary < max_len:
                last_complete_idx = boundary
            else:
                break

        # 截断
        return (tokens[:last_complete_idx+1],
                types[:last_complete_idx+1],
                positions[:last_complete_idx+1])

    def _apply_masking(self, src, token_types):
        """
        应用BERT-style masking（15%）

        对三种token类型都可能mask
        """
        masked_src = src.copy()
        masked_labels = [0] * len(src)

        # 找出可以mask的位置（排除[CLS], [SEP], [PAD]）
        maskable_positions = []
        for i, token_id in enumerate(src):
            if token_id not in [self.vocab.get(CLS_TOKEN),
                               self.vocab.get(SEP_TOKEN),
                               PAD_ID]:
                maskable_positions.append(i)

        # 随机选择15%
        num_to_mask = max(1, int(len(maskable_positions) * 0.15))
        random.shuffle(maskable_positions)
        mask_positions = maskable_positions[:num_to_mask]

        for pos in mask_positions:
            masked_labels[pos] = src[pos]  # 记录原始token

            prob = random.random()
            if prob < 0.8:
                # 80%: 替换为[MASK]
                masked_src[pos] = self.vocab.get(MASK_TOKEN)
            elif prob < 0.9:
                # 10%: 替换为随机token（根据token类型）
                token_type = token_types[pos]
                if token_type == 0:  # raw packet
                    masked_src[pos] = random.randint(0, 65535)
                elif token_type == 1:  # temporal
                    masked_src[pos] = random.randint(0, 999)
                elif token_type == 2:  # size
                    masked_src[pos] = random.randint(0, 3000)
            # 10%: 保持不变

        return masked_src, masked_labels


class MultiModalDataLoader(DataLoader):
    """多模态数据加载器"""

    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()
            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src = []
            mask_labels = []
            token_types = []
            positions = []

            for ins in instances:
                src.append(ins[0])
                mask_labels.append(ins[1])
                token_types.append(ins[2])
                positions.append(ins[3])

            yield (torch.LongTensor(src),
                   torch.LongTensor(mask_labels),
                   torch.LongTensor(token_types),
                   torch.LongTensor(positions))