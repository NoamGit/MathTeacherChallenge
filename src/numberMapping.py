import re
import pandas as pd

PRECEDING_DICT = ['number','numbers','digit','digits','integer','integers','consecutive']

def number_mapper(str, eq_num_list):
    new_str = text2int(str.lower())
    num_list = re.findall(r'[0-9/]+', new_str)
    var_list = []
    new_str = re.sub(r'[0-9/]+', '$N', new_str)
    original_eq_num_list = eq_num_list.copy()
    for possible_num in num_list:
        # real number
        if possible_num in eq_num_list:
            i = original_eq_num_list.index(possible_num)
            new_str = re.sub(r'\$N', f'$n{i}', new_str, count=1)
            var_list.append(f'$n{i}')
            eq_num_list.remove(possible_num)
        else:
            new_str = re.sub(r'\$N', f'$M', new_str, count=1)
            var_list.append(f'$v')

    # new_str = fix_string(new_str, str)
    return new_str, num_list, var_list


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
    tokenized_text = textnum.split()
    for word,word_next in zip(tokenized_text,tokenized_text[1:]+[';']):
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
            elif word_next in PRECEDING_DICT:
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
        if str[:4] == 'unkn':
            new_text_list.append(str)
            continue
        new_str = text2int(str.lower())
        num_list = re.findall(r'[0-9/]+', new_str)
        new_str = re.sub(r'[0-9/]+', '$N', new_str)

        new_text_list.append(new_str)
        numbers_list += num_list

    # count the number n0, n1, ...
    for i in range(len(numbers_list)):
        for j, equation in enumerate(new_text_list):
            new_text_list[j], is_changed = re.subn(r'\$N', f'$n{i}', equation, count=1)
            if is_changed:
                break
    return new_text_list, numbers_list


def number_parsing(equation_list, text):
    '''
    :param equation_list: list of the equations
    :param text: the word math question
    :return: equation list: a new list where numbers apear as $n0, $n1,...
            eq_num_list: values of the $n0, $n1,...
            text: the word math question with numbers as $n0, $n1,...
    '''
    text = re.sub('[.,]', '', text)
    equation_list_template, eq_num_list = list_number_mapper(equation_list)
    tmp_eq_num_list = eq_num_list.copy()
    text, text_num_list, var_list = number_mapper(text.replace("-", " "), tmp_eq_num_list)
    new_equation_list = generate_new_equation(equation_list_template, eq_num_list, var_list, text_num_list)
    return new_equation_list, eq_num_list, text, var_list, text_num_list


def test_number_parsing(text):
    text = re.sub('[.,]', '', text).replace("-", " ")
    new_text_list, numbers_list = list_number_mapper([text])
    return new_text_list[0], numbers_list


def generate_new_equation(equation_list_template, eq_num_list, var_list, text_num_list):
    new_equations = [equation_list_template[0]]
    for equation in equation_list_template[1:]:
        # put back the old numbers from the equation
        for i in range(len(eq_num_list)):
            if f'$n{i}' not in var_list:
                equation = re.sub(f'\$n{i}', eq_num_list[i], equation)

        # finish
        new_equations.append(equation)

    return new_equations


if __name__ == '__main__':
    data = pd.read_json(r'..\Data\dolphin-number_word_std\number_word_std.dev.json')
    # data = pd.read_json(r'C:\Users\Five\Documents\DataHack\Data\dolphin-number_word_std\number_word_std.test.json')

    ii = 271
    equation_list = data.iloc[ii].equations
    text = data.iloc[ii].text
    equation_list_template, eq_num_list, text_template, var_list, text_num_list = number_parsing(equation_list, text)

    print(
        f"\noriginal text:\t\t{text}\ntemplate text:\t\t{text_template}\ntext num list:\t\t{text_num_list}\nequation list:\t\t{equation_list}")
    print(
        f"eq list template:\t{equation_list_template}\nnumlist from eq:\t\t{eq_num_list}\nfinal numlist from text:\t{var_list}")

    # test the test set
    test_template, text_num_list = test_number_parsing(text)
    print(f'test_template: \t {test_template}')
    print(f'num list: \t {text_num_list}')
    # for i in range(1):
    #     test_text = test_data.iloc[i].text
    #     new_text, numbers_list = test_number_parsing(test_text)
    #     print(new_text, numbers_list)
