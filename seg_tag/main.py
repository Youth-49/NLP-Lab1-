from mpseg import MPseg
from spseg import SPseg
from pos import WordTagging
from score import Score, Score_hard

class ToolKit(MPseg, SPseg, WordTagging):
    def __init__(self, wordFile: str = 'data/wordpieces.txt', seqFile: str = 'data/sequences.txt'):
        super().__init__(wordFile=wordFile, seqFile=seqFile)

    def process(self, sentence: str, cmd: str = None, smooth: str = None):
        if (cmd is None):
            return sentence
        
        if (cmd == 'mpseg'):
            if smooth:
                return self.mpcut(sentence, smooth)
            else:
                return self.mpcut(sentence, None)

        if (cmd == 'spseg'):
            return self.spcut(sentence)

        if (cmd == 'pos'):
            return self.tagging(sentence)

        print('unknown cmd!')
        return sentence

if __name__ == '__main__':
    tk = ToolKit()
    # test sentence(demo)
    sentence = '沙曼说，何鲁丽访老是一次重要的高层次访问和交流'
    label_cut = ['沙曼', '说', '，', '何', '鲁丽', '访', '老', '是', '一', '次', '重要', '的', '高', '层次', '访问', '和', '交流']
    label_pos = ['nr', 'v', 'w', 'nr', 'nr', 'v', 'j', 'v', 'm', 'q', 'a', 'u', 'a', 'n', 'vn', 'c', 'vn']

    pred_mp = tk.process(sentence, cmd='mpseg')
    pred_mp_add1 = tk.process(sentence, cmd='mpseg', smooth='Add1') # Add1 or Jelinek-Mercer smoothing method
    pred_mp_JM = tk.process(sentence, cmd='mpseg', smooth='Jelinek-Mercer')
    pred_sp = tk.process(sentence, cmd='spseg')
    pred_pos = tk.process(label_cut, cmd='pos')

    mp_P, mp_R, mp_F = Score(pred=pred_mp, groundTrue=label_cut)
    mp_add1_P, mp_add1_R, mp_add1_F = Score(pred=pred_mp_add1, groundTrue=label_cut)
    mp_JM_P, mp_JM_R, mp_JM_F = Score(pred=pred_mp_JM, groundTrue=label_cut)
    sp_P, sp_R, sp_F = Score(pred=pred_sp, groundTrue=label_cut)
    pos_P, pos_R, pos_F = Score(pred=pred_pos, groundTrue=label_pos)
    pos_P_hard, pos_R_hard, pos_F_hard = Score_hard(pred=pred_pos, groundTrue=label_pos)
    
    print('--> Maximum probability model(Cutting):')
    print('----> no smooth:')
    print(pred_mp)
    print(f'Precision:{mp_P}, Recall:{mp_R}, F1:{mp_F}\n')
    print('----> Add1 smooth:')
    print(pred_mp_add1)
    print(f'Precision:{mp_add1_P}, Recall:{mp_add1_R}, F1:{mp_add1_F}\n')
    print('----> JM smooth:')
    print(pred_mp_JM)
    print(f'Precision:{mp_JM_P}, Recall:{mp_JM_R}, F1:{mp_JM_F}\n')

    print('--> Shortest path model(Cutting):')
    print(pred_sp)
    print(f'Precision:{sp_P}, Recall:{sp_R}, F1:{sp_F}\n')
    
    print('--> HMM model(POS):')
    print(pred_pos)
    print(f'Coarse: Precision:{pos_P}, Recall:{pos_R}, F1:{pos_F}')
    print(f'Strict: Precision:{pos_P_hard}, Recall:{pos_R_hard}, F1:{pos_F_hard}')