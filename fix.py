from NER import iob_util

if __name__ == '__main__':
    m = '-m'
    d = '-d'

    O = 'O'
    B = 'B' + m
    I = 'I' + m

    test = [O, B, I, B, I, O, O, O, O, B, I, I, I, I, O, O]
    a = iob_util.convert_iob_taglist_to_dict(test)

    print(test)
