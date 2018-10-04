import re
import pandas as pd


def number_mapper(str, eq_num_list):
    new_str = text2int(str.lower())
    num_list = re.findall(r'\d+', new_str)
    var_list = []
    new_str = re.sub(r'\d+', '$N', new_str)
    i = 0
    for possible_num in num_list:
        # real number
        if int(possible_num) in eq_num_list:
            new_str = re.sub(r'\$N', f'$n{i}', new_str, count=1)
            var_list.append(f'$n{i}')
            i += 1
            eq_num_list.remove(int(possible_num))
        else:
            new_str = re.sub(r'\$N', f'$M', new_str, count=1)
            var_list.append(f'$v')

    new_str = fix_string(new_str, str)
    return new_str, list(map(int, num_list)), var_list


def fix_string(new_str, str):
    new_str_list = new_str.split(' ')
    str_list = str.split(' ')
    for i, word in enumerate(new_str_list):
        if word == '$M':
            new_str_list[i] = str_list[i]
    return ' '.join(new_str_list)


def text2int(textnum, numwords={}):
    if not numwords:
        units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        # numwords["and"] = (1, 0)
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'twice': 2}
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', '-')

    current = result = 0
    curstring = ""
    onnumber = False
    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]

                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True

    if onnumber:
        curstring += repr(result + current)

    return curstring


def list_number_mapper(eq_list):
    new_text_list = []
    numbers_list = []
    for str in eq_list:
        new_str = text2int(str.lower())
        num_list = re.findall(r'\d+', new_str)
        new_str = re.sub(r'\d+', '$N', new_str)
        num_list = list(map(int, num_list))

        new_text_list.append(new_str)
        numbers_list += num_list

    # count the number n0, n1, ...
    for i in range(len(numbers_list)):
        for j, equation in enumerate(new_text_list):
            new_text_list[j], is_changed = re.subn(r'\$N', f'$n{i}', equation, count=1)
            if is_changed:
                break
    return new_text_list, numbers_list


def index_number(equation_list, text, i):
    for j, equation in enumerate(equation_list):
        equation_list[j], is_changed = re.subn(r'\$N', f'$n{i}', equation, count=1)
        if is_changed:
            break
    return equation_list, text


def number_parsing(equation_list, text):
    '''
    :param equation_list: list of the equations
    :param text: the word math question
    :return: equation list: a new list where numbers apear as $n0, $n1,...
            eq_num_list: values of the $n0, $n1,...
            text: the word math question with numbers as $n0, $n1,...
    '''
    equation_list, eq_num_list = list_number_mapper(equation_list)
    tmp_eq_num_list = eq_num_list.copy()
    text, _, var_list = number_mapper(text, tmp_eq_num_list)
    return equation_list, eq_num_list, text, var_list


def test_number_parsing(text):
    return list_number_mapper([text])[0]


if __name__ == '__main__':
    train_data = pd.read_json(r'C:\Users\Five\Documents\DataHack\Data\dolphin-number_word_std\number_word_std.dev.json')

    for i in range(5):
        equation_list = train_data.iloc[i].equations
        text = train_data.iloc[i].text

        print(text)
        print(equation_list)

        equation_list, eq_num_list, text, var_list = number_parsing(equation_list, text)
        print(text, '\n', equation_list, '\n', eq_num_list, '\n', var_list,'\n')

    # test the test set
    test_data = pd.read_json(
        r'C:\Users\Five\Documents\DataHack\Data\dolphin-number_word_std\number_word_std.test.json')

    for i in range(1):
        test_text = test_data.iloc[i].text
        new_text, numbers_list = test_number_parsing(test_text)
        print(new_text, numbers_list)
