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
    Dataset for Packet Size modality (Stage 1)

    Corpus format (corpus_size.txt):
    ||
    6
    1672 2185 953 ...
    ||
    17
    1300 1400 ...

    Format explanation:
    - || : Flow separator
    - Next line: Protocol (6=TCP, 17=UDP)
    - Following line: Size tokens (direction already encoded: size * direction + 1500)
    """

    def __init__(self, args, vocab, tokenizer):
        super(PacketSizeDataset, self).__init__(args, vocab, tokenizer)
        self.short_seq_prob = args.short_seq_prob

    def worker(self, proc_id, start, end):
        print("Worker %d is building packet size dataset..." % proc_id)
        set_seed(self.seed)

        # Buffer for accumulating flows
        flow_buffer = []   # List of token lists (one per flow)
        flow_proto = []    # Protocol for each flow (0=TCP, 1=UDP)

        # Current flow being parsed
        current_tokens = []
        current_protocol = 0

        pos = 0
        state = 'init'  # States: 'init', 'protocol', 'tokens'

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
                    if len(current_tokens) > 0:
                        flow_buffer.append(current_tokens)
                        flow_proto.append(current_protocol)

                    # Check if buffer is full
                    if len(flow_buffer) > self.docs_buffer_size:
                        instances = self.build_instances(flow_buffer, flow_proto)
                        for instance in instances:
                            pickle.dump(instance, dataset_writer)
                        flow_buffer = []
                        flow_proto = []

                    # Reset for new flow
                    current_tokens = []
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
                    state = 'tokens'

                elif state == 'tokens':
                    # This line contains size tokens
                    if not line:
                        continue

                    # Tokenize size tokens
                    tokens = self.tokenizer.convert_tokens_to_ids(
                        self.tokenizer.tokenize(line)
                    )

                    if len(tokens) > 0:
                        current_tokens = tokens

            # Save last flow
            if len(current_tokens) > 0:
                flow_buffer.append(current_tokens)
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
        """Create training instances from a single flow"""
        tokens = all_flows[flow_index]
        protocol = flow_proto[flow_index] if flow_index < len(flow_proto) else 0

        max_num_tokens = self.seq_length - 2  # Reserve for [CLS] and [SEP]

        # Truncate
        if len(tokens) > max_num_tokens:
            tokens = tokens[:max_num_tokens]

        # Build instance with special tokens
        src = [self.vocab.get(CLS_TOKEN)] + tokens + [self.vocab.get(SEP_TOKEN)]

        # Padding
        while len(src) < self.seq_length:
            src.append(PAD_ID)

        # Apply masking
        if not self.dynamic_masking:
            src, tgt_mlm = mask_seq(src, self.tokenizer, self.whole_word_masking,
                                   self.span_masking, self.span_geo_prob, self.span_max_length)
            instance = (src, tgt_mlm, protocol)
        else:
            instance = (src, protocol)

        return [instance]


class PacketSizeDataLoader(DataLoader):
    """DataLoader for Packet Size modality"""

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
            protocols = []

            masked_words_num = 0

            for ins in instances:
                if len(ins) == 3:  # Static masking
                    src.append(ins[0])
                    masked_words_num += len(ins[1])
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in ins[1]:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    protocols.append(ins[2])
                else:  # Dynamic masking
                    src_single, tgt_mlm_single = mask_seq(
                        ins[0], self.tokenizer, self.whole_word_masking,
                        self.span_masking, self.span_geo_prob, self.span_max_length
                    )
                    masked_words_num += len(tgt_mlm_single)
                    src.append(src_single)
                    tgt_mlm.append([0] * len(ins[0]))
                    for mask in tgt_mlm_single:
                        tgt_mlm[-1][mask[0]] = mask[1]
                    protocols.append(ins[1])

            if masked_words_num == 0:
                continue

            yield (torch.LongTensor(src),
                   torch.LongTensor(tgt_mlm),
                   torch.LongTensor(protocols))