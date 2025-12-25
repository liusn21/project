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
        tmp_dataset_reader = open("/tmp/" + str(i) + ".pt", "rb")
        while True:
            tmp_data = tmp_dataset_reader.read(2^20) 
            if tmp_data:
                dataset_writer.write(tmp_data)
            else:
                break
        tmp_dataset_reader.close()
        os.remove("/tmp/" + str(i) + ".pt")
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
# Stage 1: Single-Modal Datasets (Raw Packet & Packet Size)
# ============================================================================

class RawPacketDataset(Dataset):
    """
    Dataset for Raw Packet modality (Stage 1)

    Corpus format (corpus_raw.txt):
    ||
    6
    1 4500 0006 0683 3c52 5297 ...
    -1 4500 0000 003c 3c52 ...
    ||
    17
    1 ...
    ...

    Format explanation:
    - || : Flow separator
    - Next line: Protocol (6=TCP, 17=UDP)
    - Following lines: "{direction} {bigrams...}" per packet
    """

    def __init__(self, args, vocab, tokenizer):
        super(RawPacketDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building raw packet dataset..." % proc_id)
        set_seed(self.seed)

        # Buffer for accumulating flows
        flow_buffer = []      # List of flows, each flow is list of packets
        flow_proto = []       # Protocol for each flow (0=TCP, 1=UDP)
        flow_directions = []  # Directions for each flow

        # Current flow being parsed
        current_packets = []       # List of token lists for current flow
        current_directions = []    # List of direction values for current flow
        current_protocol = 0       # Protocol for current flow

        pos = 0
        state = 'init'  # States: 'init', 'protocol', 'packets'

        dataset_writer = open("/tmp/" + str(proc_id) + ".pt", "wb")

        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            # Skip to start position
            while pos < start:
                f.readline()
                pos += 1

            while True:
                line = f.readline()
                if not line:  # EOF
                    break
                pos += 1

                line = line.strip()

                # Check for flow separator
                if line == "||":
                    # Save previous flow if exists
                    if len(current_packets) > 0:
                        flow_buffer.append(current_packets)
                        flow_proto.append(current_protocol)
                        flow_directions.append(current_directions)

                    # Check if buffer is full
                    total_packets = sum(len(f) for f in flow_buffer)
                    if total_packets > self.docs_buffer_size:
                        instances = self.build_instances(flow_buffer, flow_proto, flow_directions)
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)
                        flow_buffer = []
                        flow_proto = []
                        flow_directions = []

                    # Reset for new flow
                    current_packets = []
                    current_directions = []
                    state = 'protocol'
                    continue

                if pos > end:
                    break

                # Parse based on state
                if state == 'protocol':
                    # This line contains protocol number (6 or 17)
                    try:
                        proto_num = int(line)
                        current_protocol = 0 if proto_num == 6 else 1  # 0=TCP, 1=UDP
                    except ValueError:
                        current_protocol = 0  # Default TCP
                    state = 'packets'

                elif state == 'packets':
                    # This line contains: "{direction} {bigrams...}"
                    if not line:
                        continue

                    parts = line.split(' ', 1)  # Split into direction and rest
                    if len(parts) < 2:
                        continue

                    try:
                        direction = int(parts[0])  # 1 or -1
                    except ValueError:
                        direction = 1

                    bigram_str = parts[1]  # "4500 0006 0683 ..."

                    # Tokenize bigrams
                    tokens = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(bigram_str)
                    )

                    if len(tokens) > 0:
                        current_packets.append(tokens)
                        current_directions.append(direction)

            # Save last flow
            if len(current_packets) > 0:
                flow_buffer.append(current_packets)
                flow_proto.append(current_protocol)
                flow_directions.append(current_directions)

            # Process remaining buffer
            if len(flow_buffer) > 0:
                instances = self.build_instances(flow_buffer, flow_proto, flow_directions)
                for instance in instances:
                    pickle.dump(instance, dataset_writer)

        dataset_writer.close()
        print("Worker %d finished." % proc_id)

    def build_instances(self, all_flows, flow_proto, flow_directions):
        """Build training instances from accumulated flows"""
        instances = []
        for _ in range(self.dup_factor):
            for flow_index in range(len(all_flows)):
                instances.extend(
                    self.create_ins_from_flow(all_flows, flow_index, flow_proto, flow_directions)
                )
        return instances

    def create_ins_from_flow(self, all_flows, flow_index, flow_proto, flow_directions):
        """
        Create training instances from a single flow

        New approach: Track packet_id (0-7) and direction (1/-1) for each token
        - packet_ids: 0-7 for packets (max 8 packets), 8 for special tokens and padding
        - directions: 1 (uplink) or -1 (downlink) for each packet
        """
        packets = all_flows[flow_index]           # List of token lists
        directions = flow_directions[flow_index]  # List of direction values (1 or -1) per packet

        max_num_tokens = self.seq_length - 2  # Reserve for [CLS] and [SEP]
        instances = []

        # Concatenate all packets into one sequence, tracking packet index and direction
        tokens = []
        packet_indices = []  # Track which packet each token belongs to (0-7)
        token_directions = []  # Track direction for each token

        for pkt_idx, packet_tokens in enumerate(packets):
            # Limit to 8 packets (indices 0-7)
            if pkt_idx >= 8:
                break

            pkt_direction = directions[pkt_idx] if pkt_idx < len(directions) else 1

            # Each token in the packet gets the same packet index and direction
            tokens.extend(packet_tokens)
            packet_indices.extend([pkt_idx] * len(packet_tokens))
            token_directions.extend([pkt_direction] * len(packet_tokens))

        # Truncate to max length
        if len(tokens) > max_num_tokens:
            tokens = tokens[:max_num_tokens]
            packet_indices = packet_indices[:max_num_tokens]
            token_directions = token_directions[:max_num_tokens]

        # Build sequence with special tokens
        src = [self.vocab.get(CLS_TOKEN)] + tokens + [self.vocab.get(SEP_TOKEN)]

        # Packet indices: 8 is used for special tokens ([CLS], [SEP]) and padding
        packet_ids = [8]  # [CLS] gets special index 8
        packet_ids.extend(packet_indices)  # Regular tokens get their packet indices (0-7)
        packet_ids.append(8)  # [SEP] gets special index 8

        # Direction: convert -1,1 to 0,2 for embedding index (1 reserved for padding/special tokens)
        # -1 -> 0 (downlink), 1 -> 2 (uplink), padding/special -> 1
        directions_seq = [1]  # [CLS] gets neutral direction
        for d in token_directions:
            directions_seq.append(0 if d == -1 else 2)  # -1->0, 1->2
        directions_seq.append(1)  # [SEP] gets neutral direction

        # Padding
        while len(src) < self.seq_length:
            src.append(PAD_ID)
            packet_ids.append(8)  # Padding gets special index 8
            directions_seq.append(1)  # Padding gets neutral direction

        # Apply masking
        if not self.dynamic_masking:
            src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking,
                                   self.span_masking, self.span_geo_prob, self.span_max_length)
            instance = (src, tgt_mlm, packet_ids, directions_seq)
        else:
            instance = (src, packet_ids, directions_seq)

        instances.append(instance)
        return instances


class RawPacketDataLoader(DataLoader):
    """
    DataLoader for Raw Packet modality

    Returns:
        src: [batch, seq_len] - token IDs
        tgt_mlm: [batch, seq_len] - MLM targets
        packet_ids: [batch, seq_len] - packet indices (0-7 for packets, 8 for special/padding)
        directions: [batch, seq_len] - direction indices (0=downlink, 1=neutral, 2=uplink)
    """

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
            packet_ids = []
            directions = []

            masked_words_num = 0

            for ins in instances:
                if len(ins) == 4:  # Static masking: (src, tgt_mlm, packet_ids, directions_seq)
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    packet_ids.append(ins[2])
                    directions.append(ins[3])
                else:  # Dynamic masking: (src, packet_ids, directions_seq)
                    src_single, tgt_mlm_single = mask_seq(
                        ins[0], self.tokenizer, self.whole_word_masking,
                        self.span_masking, self.span_geo_prob, self.span_max_length
                    )
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    packet_ids.append(ins[1])
                    directions.append(ins[2])

            if masked_words_num == 0:
                continue

            yield (torch.LongTensor(src),
                   torch.LongTensor(tgt_mlm),
                   torch.LongTensor(packet_ids),
                   torch.LongTensor(directions))


class PacketSizeDataset(Dataset):
    """
    Dataset for Packet Size modality with Temporal Information (Stage 1)

    Corpus format (corpus_size.txt):
    ||
    6
    1672 2185 953 ...
    567 123 456 ...
    ||
    17
    1300 1400 ...
    234 456 789 ...

    Format explanation:
    - || : Flow separator
    - Next line: Protocol (6=TCP, 17=UDP)
    - Line 1: Size tokens (direction already encoded: size * direction + 1500)
    - Line 2: IAT temporal tokens (0-999)
    """

    def __init__(self, args, vocab, tokenizer, vocab_temporal=None, tokenizer_temporal=None):
        super(PacketSizeDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

        # Temporal vocabulary and tokenizer
        self.vocab_temporal = vocab_temporal if vocab_temporal is not None else vocab
        self.tokenizer_temporal = tokenizer_temporal if tokenizer_temporal is not None else tokenizer

    def worker(self, proc_id, start, end):
        print("Worker %d is building packet size dataset with temporal info..." % proc_id)
        set_seed(self.seed)

        # Buffer for accumulating flows
        flow_buffer = []   # List of (size_tokens, iat_tokens) tuples (one per flow)
        flow_proto = []    # Protocol for each flow (0=TCP, 1=UDP)

        # Current flow being parsed
        current_size_tokens = []
        current_iat_tokens = []
        current_protocol = 0

        pos = 0
        state = 'init'  # States: 'init', 'protocol', 'size_tokens', 'iat_tokens'

        dataset_writer = open("/tmp/" + str(proc_id) + ".pt", "wb")

        with open(self.corpus_path, mode="r", encoding="utf-8") as f:
            # Skip to start position
            while pos < start:
                f.readline()
                pos += 1

            while True:
                line = f.readline()
                if not line:  # EOF
                    break
                pos += 1

                line = line.strip()

                # Check for flow separator
                if line == "||":
                    # Save previous flow if exists (must have both size and IAT)
                    if len(current_size_tokens) > 0 and len(current_iat_tokens) > 0:
                        flow_buffer.append((current_size_tokens, current_iat_tokens))
                        flow_proto.append(current_protocol)

                    # Check if buffer is full
                    if len(flow_buffer) > self.docs_buffer_size:
                        instances = self.build_instances(flow_buffer, flow_proto)
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)
                        flow_buffer = []
                        flow_proto = []

                    # Reset for new flow
                    current_size_tokens = []
                    current_iat_tokens = []
                    state = 'protocol'
                    continue

                if pos > end:
                    break

                # Parse based on state
                if state == 'protocol':
                    # This line contains protocol number (6 or 17)
                    try:
                        proto_num = int(line)
                        current_protocol = 0 if proto_num == 6 else 1  # 0=TCP, 1=UDP
                    except ValueError:
                        current_protocol = 0  # Default TCP
                    state = 'size_tokens'

                elif state == 'size_tokens':
                    # This line contains size tokens
                    if not line:
                        continue

                    # Tokenize size tokens
                    tokens = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(line)
                    )

                    if len(tokens) > 0:
                        current_size_tokens = tokens
                        state = 'iat_tokens'

                elif state == 'iat_tokens':
                    # This line contains IAT temporal tokens
                    if not line:
                        continue

                    # Tokenize IAT tokens
                    iat_tokens = self.tokenizer_temporal.convert_tokens_to_ids(
                        self.tokenizer_temporal.tokenize(line)
                    )

                    if len(iat_tokens) > 0:
                        current_iat_tokens = iat_tokens
                    # After reading IAT, flow is complete, wait for next separator

            # Save last flow
            if len(current_size_tokens) > 0 and len(current_iat_tokens) > 0:
                flow_buffer.append((current_size_tokens, current_iat_tokens))
                flow_proto.append(current_protocol)

            # Process remaining buffer
            if len(flow_buffer) > 0:
                instances = self.build_instances(flow_buffer, flow_proto)
                for instance in instances:
                    pickle.dump(instance, dataset_writer)

        dataset_writer.close()
        print("Worker %d finished." % proc_id)

    def build_instances(self, all_flows, flow_proto):
        """Build training instances from accumulated flows"""
        instances = []
        for _ in range(self.dup_factor):
            for flow_index in range(len(all_flows)):
                instances.extend(self.create_ins_from_flow(all_flows, flow_index, flow_proto))
        return instances

    def create_ins_from_flow(self, all_flows, flow_index, flow_proto):
        """
        Create training instances from a single flow with temporal information

        Each flow contains (size_tokens, iat_tokens)
        Both modalities are masked independently with the same mask positions
        """
        size_tokens, iat_tokens = all_flows[flow_index]
        # flow_proto is ignored as per requirements

        # Ensure size and IAT have same length
        min_len = min(len(size_tokens), len(iat_tokens))
        size_tokens = size_tokens[:min_len]
        iat_tokens = iat_tokens[:min_len]

        max_num_tokens = self.seq_length - 2  # Reserve for [CLS] and [SEP]

        # Truncate
        if len(size_tokens) > max_num_tokens:
            size_tokens = size_tokens[:max_num_tokens]
            iat_tokens = iat_tokens[:max_num_tokens]

        # Build instance with special tokens for both modalities
        src_size = [self.vocab.get(CLS_TOKEN)] + size_tokens + [self.vocab.get(SEP_TOKEN)]
        src_iat = [self.vocab_temporal.get(CLS_TOKEN)] + iat_tokens + [self.vocab_temporal.get(SEP_TOKEN)]

        # Padding
        while len(src_size) < self.seq_length:
            src_size.append(PAD_ID)
            src_iat.append(PAD_ID)

        # Apply masking (use same mask positions for both modalities)
        if not self.dynamic_masking:
            # Mask size tokens
            src_size_masked, tgt_mlm_size = mask_seq(
                src_size, self.tokenizer, self.whole_word_masking,
                self.span_masking, self.span_geo_prob, self.span_max_length
            )

            # Mask IAT tokens at the same positions with 80/10/10 strategy
            # Extract mask positions from size masking
            mask_positions = [pos for pos, _ in tgt_mlm_size]

            # Apply same mask positions to IAT with standard BERT masking strategy
            src_iat_masked = src_iat.copy()
            tgt_mlm_iat = []
            for pos in mask_positions:
                original_token = src_iat[pos]
                tgt_mlm_iat.append((pos, original_token))

                # Standard BERT masking: 80% [MASK], 10% random, 10% unchanged
                prob = random.random()
                if prob < 0.8:
                    # 80%: Replace with [MASK]
                    src_iat_masked[pos] = self.vocab_temporal.get(MASK_TOKEN)
                elif prob < 0.9:
                    # 10%: Replace with random token
                    while True:
                        rdi = random.randint(1, len(self.vocab_temporal) - 1)
                        if rdi not in [self.vocab_temporal.get(CLS_TOKEN),
                                      self.vocab_temporal.get(SEP_TOKEN),
                                      self.vocab_temporal.get(MASK_TOKEN),
                                      PAD_ID]:
                            break
                    src_iat_masked[pos] = rdi
                # else: 10%: Keep original token (src_iat_masked already has it)

            instance = (src_size_masked, src_iat_masked, tgt_mlm_size, tgt_mlm_iat)
        else:
            instance = (src_size, src_iat)

        return [instance]


class PacketSizeDataLoader(DataLoader):
    """
    DataLoader for Packet Size modality with Temporal Information

    Returns:
        src_size: [batch, seq_len] - size token IDs (direction encoded)
        src_iat: [batch, seq_len] - IAT temporal token IDs
        tgt_mlm_size: [batch, seq_len] - size MLM targets
        tgt_mlm_temporal: [batch, seq_len] - temporal MLM targets
    """

    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=False):
        super(PacketSizeDataLoader, self).__init__(args, dataset_path, batch_size, proc_id, proc_num, shuffle)
        # Get temporal tokenizer from args (if available)
        self.tokenizer_temporal = getattr(args, 'tokenizer_temporal', self.tokenizer)
        self.vocab_temporal = getattr(args, 'vocab_temporal', self.vocab)

    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()

            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            src_size = []
            src_iat = []
            tgt_mlm_size = []
            tgt_mlm_temporal = []

            masked_words_num = 0

            for ins in instances:
                if len(ins) == 4:  # Static masking: (src_size, src_iat, tgt_mlm_size, tgt_mlm_iat)
                    src_size.append(ins[0])
                    src_iat.append(ins[1])
                    masked_words_num += len(ins[2])  # Count size masks

                    # Convert mask tuples to dense targets
                    tgt_mlm_size.append([0] * len(ins[0]))
                    for mask in ins[2]:
                        tgt_mlm_size[-1][mask[0]] = mask[1]

                    tgt_mlm_temporal.append([0] * len(ins[1]))
                    for mask in ins[3]:
                        tgt_mlm_temporal[-1][mask[0]] = mask[1]

                else:  # Dynamic masking: (src_size, src_iat)
                    # Mask size tokens
                    src_size_single, tgt_mlm_size_single = mask_seq(
                        ins[0], self.tokenizer, self.whole_word_masking,
                        self.span_masking, self.span_geo_prob, self.span_max_length
                    )
                    masked_words_num += len(tgt_mlm_size_single)

                    # Mask IAT tokens at the same positions with 80/10/10 strategy
                    mask_positions = [pos for pos, _ in tgt_mlm_size_single]
                    src_iat_single = ins[1].copy()
                    tgt_mlm_iat_single = []
                    for pos in mask_positions:
                        original_token = ins[1][pos]
                        tgt_mlm_iat_single.append((pos, original_token))

                        # Standard BERT masking: 80% [MASK], 10% random, 10% unchanged
                        prob = random.random()
                        if prob < 0.8:
                            # 80%: Replace with [MASK]
                            src_iat_single[pos] = self.vocab_temporal.get(MASK_TOKEN)
                        elif prob < 0.9:
                            # 10%: Replace with random token
                            while True:
                                rdi = random.randint(1, len(self.vocab_temporal) - 1)
                                if rdi not in [self.vocab_temporal.get(CLS_TOKEN),
                                              self.vocab_temporal.get(SEP_TOKEN),
                                              self.vocab_temporal.get(MASK_TOKEN),
                                              PAD_ID]:
                                    break
                            src_iat_single[pos] = rdi
                        # else: 10%: Keep original token

                    src_size.append(src_size_single)
                    src_iat.append(src_iat_single)

                    # Convert to dense targets
                    tgt_mlm_size.append([0] * len(ins[0]))
                    for mask in tgt_mlm_size_single:
                        tgt_mlm_size[-1][mask[0]] = mask[1]

                    tgt_mlm_temporal.append([0] * len(ins[1]))
                    for mask in tgt_mlm_iat_single:
                        tgt_mlm_temporal[-1][mask[0]] = mask[1]

            if masked_words_num == 0:
                continue

            yield (torch.LongTensor(src_size),
                   torch.LongTensor(src_iat),
                   torch.LongTensor(tgt_mlm_size),
                   torch.LongTensor(tgt_mlm_temporal))


# ============================================================================
# Stage 2: Multi-Modal Dataset (Raw Packet + Packet Size)
# ============================================================================

class MultiModalDataset(Dataset):
    """
    Dataset for Multi-Modal Pretraining (Stage 2)

    Loads paired Raw Packet + Packet Size data for CMM and CMMP tasks

    Requires:
        - corpus_path_raw: Path to raw packet corpus
        - corpus_path_size: Path to packet size corpus

    Both corpora must have the same flows in the same order.
    """

    def __init__(self, args, vocab_raw, vocab_size, tokenizer_raw, tokenizer_size):
        # Initialize base class with raw vocab/tokenizer (for compatibility)
        super(MultiModalDataset, self).__init__(args, vocab_raw, tokenizer_raw)

        # Store both vocabularies and tokenizers
        self.vocab_raw = vocab_raw
        self.vocab_size = vocab_size
        self.tokenizer_raw = tokenizer_raw
        self.tokenizer_size = tokenizer_size

        # Corpus paths
        self.corpus_path_raw = args.corpus_path_raw
        self.corpus_path_size = args.corpus_path_size

        # Sequence lengths (separate for each modality)
        self.seq_length_raw = getattr(args, 'seq_length_raw', 512)
        self.seq_length_size = getattr(args, 'seq_length_size', 256)

        self.short_seq_prob = args.short_seq_prob if hasattr(args, 'short_seq_prob') else 0.0

    def build_and_save(self, workers_num):
        """
        Build multi-modal dataset from paired corpus files

        Overrides base class to handle two corpus files.
        Split by flow count (not line count).
        """
        # Count flows in both corpora (by counting '||' separators)
        print("Counting flows in corpora...")
        flow_count_raw = self._count_flows(self.corpus_path_raw)
        flow_count_size = self._count_flows(self.corpus_path_size)

        print(f"Raw corpus: {flow_count_raw} flows")
        print(f"Size corpus: {flow_count_size} flows")

        if flow_count_raw != flow_count_size:
            print(f"WARNING: Flow counts don't match! Using minimum: {min(flow_count_raw, flow_count_size)}")
            total_flows = min(flow_count_raw, flow_count_size)
        else:
            total_flows = flow_count_raw

        print(f"Starting {workers_num} workers for building multi-modal dataset...")
        print(f"Total flows to process: {total_flows}")
        assert workers_num >= 1

        if workers_num == 1:
            self.worker(0, 0, total_flows)
        else:
            from multiprocessing import Pool
            pool = Pool(workers_num)

            # Split work evenly by flow count
            for i in range(workers_num):
                start_flow = i * total_flows // workers_num
                end_flow = (i + 1) * total_flows // workers_num
                pool.apply_async(func=self.worker, args=[i, start_flow, end_flow])

            pool.close()
            pool.join()

        # Merge datasets
        merge_dataset(self.dataset_path, workers_num)
        print(f"Dataset saved to: {self.dataset_path}")

    def _count_flows(self, corpus_path):
        """Count number of flows (by counting '||' separators)"""
        count = 0
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() == '||':
                    count += 1
        return count

    def worker(self, proc_id, start_flow, end_flow):
        """
        Worker process to build dataset

        Args:
            proc_id: Process ID
            start_flow: Starting flow index (inclusive)
            end_flow: Ending flow index (exclusive)
        """
        print(f"Worker {proc_id} processing flows {start_flow} to {end_flow-1}...")
        set_seed(self.seed)

        # Buffers for both modalities
        raw_flow_buffer = []
        raw_proto_buffer = []
        raw_directions_buffer = []

        size_flow_buffer = []
        size_proto_buffer = []

        dataset_writer = open("/tmp/" + str(proc_id) + ".pt", "wb")

        # Open both corpus files
        with open(self.corpus_path_raw, mode="r", encoding="utf-8") as f_raw, \
             open(self.corpus_path_size, mode="r", encoding="utf-8") as f_size:

            # Skip to start_flow in both files
            if start_flow > 0:
                self._skip_flows(f_raw, start_flow)
                self._skip_flows(f_size, start_flow)

            # Process flows from start_flow to end_flow
            flows_processed = 0
            for flow_idx in range(start_flow, end_flow):
                # Parse one flow from Raw corpus
                raw_packets, raw_proto, raw_directions = self._parse_raw_flow(f_raw)
                if raw_packets is None:
                    print(f"Worker {proc_id}: Reached end of raw corpus at flow {flow_idx}")
                    break

                # Parse one flow from Size corpus
                size_tokens, size_proto = self._parse_size_flow(f_size)
                if size_tokens is None:
                    print(f"Worker {proc_id}: Reached end of size corpus at flow {flow_idx}")
                    break

                # Verify protocols match
                if raw_proto != size_proto:
                    print(f"Worker {proc_id} WARNING: Protocol mismatch at flow {flow_idx}: raw={raw_proto}, size={size_proto}")

                # Add to buffers
                raw_flow_buffer.append(raw_packets)
                raw_proto_buffer.append(raw_proto)
                raw_directions_buffer.append(raw_directions)

                size_flow_buffer.append(size_tokens)
                size_proto_buffer.append(size_proto)

                flows_processed += 1

                # Check if buffer is full
                if len(raw_flow_buffer) >= self.docs_buffer_size:
                    instances = self.build_instances(
                        raw_flow_buffer, raw_proto_buffer, raw_directions_buffer,
                        size_flow_buffer, size_proto_buffer
                    )
                    for instance in instances:
                        pickle.dump(instance, dataset_writer)

                    # Clear buffers
                    raw_flow_buffer = []
                    raw_proto_buffer = []
                    raw_directions_buffer = []
                    size_flow_buffer = []
                    size_proto_buffer = []

            # Process remaining buffer
            if len(raw_flow_buffer) > 0:
                instances = self.build_instances(
                    raw_flow_buffer, raw_proto_buffer, raw_directions_buffer,
                    size_flow_buffer, size_proto_buffer
                )
                for instance in instances:
                    pickle.dump(instance, dataset_writer)

        dataset_writer.close()
        print(f"Worker {proc_id} finished processing {flows_processed} flows.")

    def _skip_flows(self, file_handle, num_flows):
        """Skip num_flows in the corpus file"""
        flows_skipped = 0
        while flows_skipped < num_flows:
            line = file_handle.readline()
            if not line:  # EOF
                break
            if line.strip() == '||':
                flows_skipped += 1

    def _parse_raw_flow(self, f):
        """
        Parse one complete flow from raw packet corpus

        Format:
            ||
            6 (or 17)
            1 4500 0006 0683 ...
            -1 4500 0000 003c ...
            ... (more packets)

        Returns:
            (packets, protocol, directions) or (None, None, None) if EOF
        """
        # Read flow separator
        line = f.readline()
        if not line:  # EOF
            return None, None, None

        line = line.strip()
        if line != "||":
            # Skip until we find a flow separator
            while line and line != "||":
                line = f.readline().strip()
            if not line:
                return None, None, None

        # Read protocol
        proto_line = f.readline()
        if not proto_line:
            return None, None, None

        proto_num = int(proto_line.strip())
        protocol = 0 if proto_num == 6 else 1  # 0=TCP, 1=UDP]

        # Read packets until next flow separator or EOF
        packets = []
        directions = []

        while True:
            # Save current position
            current_pos = f.tell()
            line = f.readline()

            if not line:  # EOF
                break

            line = line.strip()

            # Check if next flow starts
            if line == "||":
                # Put back the separator for next flow
                f.seek(current_pos)
                break

            if not line:  # Empty line, skip
                continue

            # Parse packet line: "direction bigrams..."
            parts = line.split(' ', 1)
            if len(parts) < 2:
                continue

            try:
                direction = int(parts[0])  # 1 or -1
            except:
                direction = 1  # Default uplink

            bigram_str = parts[1]

            # Tokenize bigrams
            tokens = self.tokenizer_raw.convert_tokens_to_ids(
                self.tokenizer_raw.tokenize(bigram_str)
            )

            if len(tokens) > 0:
                packets.append(tokens)
                directions.append(direction)

        return packets, protocol, directions

    def _parse_size_flow(self, f):
        """
        Parse one complete flow from packet size corpus

        Format:
            ||
            6 (or 17)
            1672 2185 953 ...

        Returns:
            (tokens, protocol) or (None, None) if EOF
        """
        # Read flow separator
        line = f.readline()
        if not line:  # EOF
            return None, None

        line = line.strip()
        if line != "||":
            # Skip until we find a flow separator
            while line and line != "||":
                line = f.readline().strip()
            if not line:
                return None, None

        # Read protocol
        proto_line = f.readline()
        if not proto_line:
            print("Warning: Unexpected EOF when reading protocol")
            return None, None

        try:
            proto_num = int(proto_line.strip())
            protocol = 0 if proto_num == 6 else 1  # 0=TCP, 1=UDP
        except:
            protocol = 0  # Default TCP

        # Read size tokens (single line)
        tokens_line = f.readline()
        if not tokens_line:
            print("Warning: Unexpected EOF when reading size tokens")
            return [], protocol

        tokens_line = tokens_line.strip()
        if not tokens_line:
            print("Warning: Empty size tokens line")
            return [], protocol

        # Tokenize size tokens
        tokens = self.tokenizer_size.convert_tokens_to_ids(
            self.tokenizer_size.tokenize(tokens_line)
        )

        return tokens, protocol

    def build_instances(self, raw_flows, raw_protos, raw_directions_list,
                       size_flows, size_protos):
        """Build paired instances from both modalities"""
        instances = []

        for _ in range(self.dup_factor):
            for flow_idx in range(len(raw_flows)):
                instance = self.create_ins_from_paired_flow(
                    raw_flows[flow_idx], raw_protos[flow_idx], raw_directions_list[flow_idx],
                    size_flows[flow_idx], size_protos[flow_idx]
                )
                if instance is not None:
                    instances.append(instance)

        return instances

    def create_ins_from_paired_flow(self, raw_packets, raw_proto, raw_directions,
                                     size_tokens, size_proto):
        """
        Create one instance from paired Raw + Size flow

        Returns:
            Static masking: (raw_src, raw_packet_ids, raw_directions_seq, size_src, tgt_cmmp_raw, tgt_cmmp_size)
            Dynamic masking: (raw_src, raw_packet_ids, raw_directions_seq, size_src)

        Note: Both Raw and Size modalities are masked for CMMP tasks
        """
        max_raw_tokens = self.seq_length_raw - 2  # Reserve for [CLS] and [SEP]
        max_size_tokens = self.seq_length_size - 2  # Reserve for [CLS] and [SEP]

        # ===== Process Raw Packet =====
        raw_tokens = []
        raw_pkt_indices = []
        raw_dir_values = []

        for pkt_idx, pkt_tokens in enumerate(raw_packets):
            if pkt_idx >= 8:  # Limit to 8 packets
                break

            pkt_direction = raw_directions[pkt_idx] if pkt_idx < len(raw_directions) else 1

            raw_tokens.extend(pkt_tokens)
            raw_pkt_indices.extend([pkt_idx] * len(pkt_tokens))
            raw_dir_values.extend([pkt_direction] * len(pkt_tokens))

        # Truncate
        if len(raw_tokens) > max_raw_tokens:
            raw_tokens = raw_tokens[:max_raw_tokens]
            raw_pkt_indices = raw_pkt_indices[:max_raw_tokens]
            raw_dir_values = raw_dir_values[:max_raw_tokens]

        # Build Raw sequence
        raw_src = [self.vocab_raw.get(CLS_TOKEN)] + raw_tokens + [self.vocab_raw.get(SEP_TOKEN)]

        raw_packet_ids = [8] + raw_pkt_indices + [8]

        raw_directions_seq = [1]  # [CLS] neutral
        for d in raw_dir_values:
            raw_directions_seq.append(0 if d == -1 else 2)
        raw_directions_seq.append(1)  # [SEP] neutral

        # Padding to seq_length_raw
        while len(raw_src) < self.seq_length_raw:
            raw_src.append(PAD_ID)
            raw_packet_ids.append(8)
            raw_directions_seq.append(1)

        # ===== Process Packet Size =====
        # Truncate
        if len(size_tokens) > max_size_tokens:
            size_tokens = size_tokens[:max_size_tokens]

        # Build Size sequence
        size_src = [self.vocab_size.get(CLS_TOKEN)] + size_tokens + [self.vocab_size.get(SEP_TOKEN)]

        # Padding to seq_length_size
        while len(size_src) < self.seq_length_size:
            size_src.append(PAD_ID)

        # ===== Apply Masking to Both Modalities =====
        if not self.dynamic_masking:
            # Static masking: apply mask_seq() here in Dataset
            
            # Mask Raw (for CMMP_raw task)
            raw_src, tgt_mlm_raw = mask_seq(
                raw_src, self.tokenizer_raw, self.whole_word_masking,
                self.span_masking, self.span_geo_prob, self.span_max_length
            )

            # Mask Size (for CMMP_size task)
            size_src, tgt_mlm_size = mask_seq(
                size_src, self.tokenizer_size, self.whole_word_masking,
                self.span_masking, self.span_geo_prob, self.span_max_length
            )

            # Return 6 elements (with both tgt_mlm_raw and tgt_mlm_size)
            return (raw_src, raw_packet_ids, raw_directions_seq, size_src, tgt_mlm_raw, tgt_mlm_size)
        else:
            # Dynamic masking: defer masking to DataLoader
            # Return 4 elements (without tgt_mlm)
            return (raw_src, raw_packet_ids, raw_directions_seq, size_src)


class MultiModalDataLoader(DataLoader):
    """
    DataLoader for Multi-Modal Pretraining (Stage 2)

    Returns positive (matching) samples only.
    CMM negative sampling and hard negative mining are done in Trainer.

    Supports both static and dynamic masking (controlled by args.dynamic_masking).

    Returns:
        raw_src: [batch, 512] - Raw Packet tokens (masked for CMMP_raw)
        raw_packet_ids: [batch, 512] - Packet indices
        raw_directions: [batch, 512] - Direction indices
        size_src: [batch, 256] - Size tokens (masked for CMMP_size)
        tgt_cmmp_raw: [batch, 512] - CMMP_raw targets (0 for unmasked, token_id for masked)
        tgt_cmmp_size: [batch, 256] - CMMP_size targets (0 for unmasked, token_id for masked)
    """

    def __init__(self, args, dataset_path, batch_size, proc_id, proc_num, shuffle=False):
        super(MultiModalDataLoader, self).__init__(
            args, dataset_path, batch_size, proc_id, proc_num, shuffle
        )
        # Store both vocabs and tokenizers for masking
        self.vocab_raw = args.vocab_raw
        self.vocab_size = args.vocab_size
        self.tokenizer_raw = args.tokenizer_raw
        self.tokenizer_size = args.tokenizer_size

    def __iter__(self):
        while True:
            while self._empty():
                self._fill_buf()

            if self.start + self.batch_size >= self.end:
                instances = self.buffer[self.start:]
            else:
                instances = self.buffer[self.start: self.start + self.batch_size]

            self.start += self.batch_size

            # Batch lists
            raw_src_batch = []
            raw_packet_ids_batch = []
            raw_directions_batch = []
            size_src_batch = []
            tgt_cmmp_raw_batch = []
            tgt_cmmp_size_batch = []

            masked_words_num = 0

            for ins in instances:
                # Instance format depends on masking mode:
                # Static masking (len=6): (raw_src, raw_packet_ids, raw_directions_seq, size_src, tgt_mlm_raw, tgt_mlm_size)
                # Dynamic masking (len=4): (raw_src, raw_packet_ids, raw_directions_seq, size_src)

                if len(ins) == 6:
                    # Static masking: Dataset already applied masking
                    raw_src = ins[0]
                    tgt_mlm_raw = ins[4]  # List of (position, token) tuples
                    size_src = ins[3]
                    tgt_mlm_size = ins[5]  # List of (position, token) tuples

                    masked_words_num += len(tgt_mlm_raw) + len(tgt_mlm_size)

                    # Convert tgt_mlm format for raw: [(pos, token), ...] -> [0, 0, token, 0, ...]
                    tgt_cmmp_raw = [0] * len(raw_src)
                    for pos, token in tgt_mlm_raw:
                        tgt_cmmp_raw[pos] = token

                    # Convert tgt_mlm format for size: [(pos, token), ...] -> [0, 0, token, 0, ...]
                    tgt_cmmp_size = [0] * len(size_src)
                    for pos, token in tgt_mlm_size:
                        tgt_cmmp_size[pos] = token

                    raw_src_batch.append(raw_src)
                    raw_packet_ids_batch.append(ins[1])
                    raw_directions_batch.append(ins[2])
                    size_src_batch.append(size_src)
                    tgt_cmmp_raw_batch.append(tgt_cmmp_raw)
                    tgt_cmmp_size_batch.append(tgt_cmmp_size)

                else:  # len(ins) == 4
                    # Dynamic masking: Apply mask_seq() here in DataLoader
                    raw_src = list(ins[0])  # Copy to avoid modifying original
                    size_src = list(ins[3])

                    # Mask raw
                    raw_src_masked, tgt_mlm_raw = mask_seq(
                        raw_src,
                        self.tokenizer_raw,
                        self.whole_word_masking,
                        self.span_masking,
                        self.span_geo_prob,
                        self.span_max_length
                    )

                    # Mask size
                    size_src_masked, tgt_mlm_size = mask_seq(
                        size_src,
                        self.tokenizer_size,
                        self.whole_word_masking,
                        self.span_masking,
                        self.span_geo_prob,
                        self.span_max_length
                    )

                    masked_words_num += len(tgt_mlm_raw) + len(tgt_mlm_size)

                    # Convert tgt_mlm format for raw
                    tgt_cmmp_raw = [0] * len(raw_src)
                    for pos, token in tgt_mlm_raw:
                        tgt_cmmp_raw[pos] = token

                    # Convert tgt_mlm format for size
                    tgt_cmmp_size = [0] * len(size_src)
                    for pos, token in tgt_mlm_size:
                        tgt_cmmp_size[pos] = token

                    raw_src_batch.append(raw_src_masked)
                    raw_packet_ids_batch.append(ins[1])
                    raw_directions_batch.append(ins[2])
                    size_src_batch.append(size_src_masked)
                    tgt_cmmp_raw_batch.append(tgt_cmmp_raw)
                    tgt_cmmp_size_batch.append(tgt_cmmp_size)

            # Skip batch if no masked words (should rarely happen)
            if masked_words_num == 0:
                continue

            yield (torch.LongTensor(raw_src_batch),
                   torch.LongTensor(raw_packet_ids_batch),
                   torch.LongTensor(raw_directions_batch),
                   torch.LongTensor(size_src_batch),
                   torch.LongTensor(tgt_cmmp_raw_batch),
                   torch.LongTensor(tgt_cmmp_size_batch))