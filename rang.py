GENDER_LIST = ['M', 'F']
AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
AGE2INDEX = {a: i for i, a in enumerate(AGE_LIST)}
GENDER2INDEX = {'M': 0, 'F': 1}


def sort_questions(age, gender, questions):
    return sorted(
        questions,
        key=lambda x:
            1 * abs(AGE2INDEX[x['rule']['age']] - AGE2INDEX[age])
            + 3 * float(x['rule']['gender'] == gender)
    )

