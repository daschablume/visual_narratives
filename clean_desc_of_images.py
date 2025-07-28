import re


def clean_text(text: str) -> dict:
    split_patt = re.compile(r'\|')
    number_patt = re.compile(r'\d{1,3}')
    llist = text.split('\n')
    image_id2desc = {}
    for string in llist:
        if not string:
            continue
        idd, desc = re.split(split_patt, string)
        idd = int(re.search(number_patt, idd).group())
        desc = desc.replace(']', '')
        image_id2desc[idd] = desc

    return image_id2desc