import task_scheduler.database.db_psql
import re
from collections import defaultdict
import task_scheduler.constant
from task_scheduler import logging_config
from task_scheduler.constant import Dictionaries

log = logging_config.getLogger()


def generate_tasks_ace(can_size):
    mentions = get_ace_mentions()
    for men in mentions:
        get_men_entities(men, 'ace2004_wned', can_size)

    with open('candidate_ents_50.csv', 'w', encoding='utf-8') as f:
        for ent in set(Dictionaries.candidate_ents):
            f.write(ent + '\n')
            # generate tasks for conll training dataset
            # if men not in Dictionaries.processed_mention:
            #     get_men_entities(men, 'train')


def generate_tasks_msnbc(can_size):
    mentions = get_msnbc_mentions()
    for men in mentions:
        get_men_entities(men, 'msnbc_new', can_size)

    with open('candidate_ents_50.csv', 'w', encoding='utf-8') as f:
        for ent in set(Dictionaries.candidate_ents):
            f.write(ent + '\n')


def generate_tasks_clueweb(can_size):
    mentions = get_clueweb_mentions()
    for men in mentions:
        get_men_entities(men, 'clueweb12', can_size)

    with open('candidate_ents_50.csv', 'w', encoding='utf-8') as f:
        for ent in set(Dictionaries.candidate_ents):
            f.write(ent + '\n')


def generate_tasks_wiki_wned(can_size):
    mentions = get_wiki_wned_mentions()
    for men in mentions:
        get_men_entities(men, 'wiki_wned', can_size)

    with open('candidate_ents_50.csv', 'w', encoding='utf-8') as f:
        for ent in set(Dictionaries.candidate_ents):
            f.write(ent + '\n')


def generate_tasks_wiki_uiuc(can_size):
    mentions = get_wiki_uiuc_mentions()
    mentions = set(mentions)
    for men_idx, men in enumerate(mentions):
        if not men_idx % 1000:
            print("Processing progress {}/{}".format(men_idx, len(mentions)))
        get_men_entities(men, 'wiki_uiuc', can_size)

    with open('candidate_ents_50.csv', 'w', encoding='utf-8') as f:
        for ent in set(Dictionaries.candidate_ents):
            f.write(ent + '\n')


def generate_tasks_kore(can_size):
    mentions = get_kore_mentions()
    mentions = set(mentions)
    for men_idx, men in enumerate(mentions):
        if not men_idx % 1000:
            print("Processing progress {}/{}".format(men_idx, len(mentions)))
        get_men_entities(men, 'kore', can_size)

    with open('candidate_ents_50.csv', 'w') as f:
        for ent in set(Dictionaries.candidate_ents):
            f.write(ent + '\n')

def generate_tasks_rss(can_size):
    mentions = get_rss_mentions()
    for men in mentions:
        get_men_entities(men, 'rss', can_size)

    with open('candidate_ents_50.csv', 'w', encoding='utf-8') as f:
        for ent in set(Dictionaries.candidate_ents):
            f.write(ent + '\n')


def generate_tasks_aquaint(can_size):
    mentions = get_aquaint_mentions()
    for men in mentions:
        get_men_entities(men, 'aquaint_new', can_size)

    with open('candidate_ents_50.csv', 'w', encoding='utf-8') as f:
        for ent in set(Dictionaries.candidate_ents):
            f.write(ent + '\n')


def generate_tasks(can_size):
    mentions = get_conll_testb_mentions()
    for men in set(mentions):
        get_men_entities(men, 'aida_conll', can_size)

    with open('candidate_ents_50.csv', 'w', encoding='utf-8') as f:
        for ent in set(Dictionaries.candidate_ents):
            f.write(ent + '\n')
            # generate tasks for conll training dataset
            # if men not in Dictionaries.processed_mention:
            #     get_men_entities(men, 'train')


# Generate tasks for left 427 queries.
def generate_tasks_sup(can_size):
    mentions = get_conll_testb_mentions_by_ids()
    for men in mentions:
        get_men_entities(men, 'aida_conll_sup', can_size)


def save_tasks_batch(entity_sets_dict):
    for key, val in entity_sets_dict.items():
        tasks = []
        for ent in val:
            tasks.append((ent, key))
        if tasks:
            Dictionaries.db.save_new_tasks_batch(tasks)
            # print(val)


def get_kore_mentions():
    mentions_kore = Dictionaries.db.fetch_mentions_kore()
    mentions = []
    for mention in mentions_kore:
        mentions.append(mention[0])
    return mentions


def get_ace_mentions():
    mentions = Dictionaries.db.fetch_all_mention_ace_non_nil()
    mentions = [mention[0] for mention in mentions]
    return mentions


def get_msnbc_mentions():
    mentions = Dictionaries.db.fetch_all_mention_msnbc_non_nil()
    mentions = [mention[0] for mention in mentions]
    return mentions


def get_clueweb_mentions():
    mentions = Dictionaries.db.fetch_all_mention_clueweb_non_nil()
    mentions = [mention[0] for mention in mentions]
    return mentions


def get_rss_mentions():
    mentions = Dictionaries.db.fetch_all_mention_rss_non_nil()
    mentions = [mention[0] for mention in mentions]
    return mentions


def get_wiki_wned_mentions():
    mentions = Dictionaries.db.fetch_all_mention_wiki_wned_non_nil()
    mentions = [mention[0] for mention in mentions]
    return mentions


def get_aquaint_mentions():
    mentions = Dictionaries.db.fetch_all_mention_aquaint_non_nil()
    mentions = [mention[0] for mention in mentions]
    return mentions


def get_wiki_uiuc_mentions():
    mentions = Dictionaries.db.fetch_all_mention_uiuc_wiki_non_nil()
    mentions = [mention[0] for mention in mentions]
    return mentions


def get_conll_testb_mentions():
    mentions_conll = Dictionaries.db.fetch_all_mention_conll_non_nil_testb()
    mentions = [mention[0] for mention in mentions_conll]
    return mentions


# Generate tasks by query ids
def get_conll_testb_mentions_by_ids():
    ids = read_ids_from_file()
    mentions_conll = Dictionaries.db.fetch_all_mention_conll_non_nil_testb_by_ids(ids)
    mentions = [mention[0] for mention in mentions_conll]
    return mentions


def escape_entity(entity):
    return str.encode(entity).decode('unicode-escape')


def get_men_entities(mention, source, can_size):
    n_men, entities_tuples = preprocess_mention(mention)
    if n_men != mention:
        log.info("{} -----> {}".format(mention, n_men))
    if entities_tuples:
        tasks = [entity[0] for entity in entities_tuples]
        # log.info("Process mention: {}".format(mention))
        selected_tasks = tasks[:min(can_size, len(tasks))] if can_size else tasks
        #log.info('%d->%d', len(tasks), len(selected_tasks))
        Dictionaries.candidate_ents.extend(selected_tasks)

    # Append ground truths
    #row = Dictionaries.db.fetch_ground_truth_entity_by_mention(source, mention)
    #if row:
    #    Dictionaries.candidate_ents.append(escape_entity(row[2]))


def normalize_form(mention):
    return mention if len(mention) < 4 else str(mention).upper()


def read_ids_from_file():
    with open('/Users/dzs/PycharmProjects/nerd-task-generation-research/no_entity_keyphrase.conll') as f:
        ids = f.readlines()
        ids = [int(e.strip()) for e in ids]
    return tuple(ids)


def split_in_words(inputstr):
    pattern = re.compile(r'(\w+)')
    words = [word for word in re.findall(pattern, inputstr)]
    return words


def first_letter_to_uppercase(s):
    return s[:1].upper() + s[1:]


def modify_uppercase_phrase(s):
    if s == s.upper():
        words = split_in_words(s.lower())
        res = [first_letter_to_uppercase(w) for w in words]
        return ' '.join(res)
    else:
        return s


def preprocess_mention(m):
    """
    Get normalized mentions and all its candidate entities.
    :param m:
    :param can_size
    :return:
    """
    cur_m = modify_uppercase_phrase(m)

    can_ents_cur_m = Dictionaries.prior_db_conn_obj. \
        fetch_entity_by_mention_size_emnlp17(cur_m, size=0)
    if cur_m == m:
        can_ents_m = can_ents_cur_m
    else:
        can_ents_m = Dictionaries.prior_db_conn_obj.fetch_entity_by_mention_size_emnlp17(m, size=0)

    mention_total_freq_m = 0 if not can_ents_m else len(can_ents_m)
    mention_total_freq_cur_m = 0 if not can_ents_cur_m else len(can_ents_cur_m)

    if not can_ents_cur_m:
        cur_m = m
        can_ents_cur_m = can_ents_m
    elif mention_total_freq_m and mention_total_freq_m > mention_total_freq_cur_m:
        cur_m = m
        can_ents_cur_m = can_ents_m

    # If we cannot find the exact mention in our index,
    # we try our luck to find it in a case insensitive index.
    men_relaxed_form = cur_m.lower()
    if not can_ents_cur_m and men_relaxed_form:
        cur_m = men_relaxed_form
        can_ents_cur_m = Dictionaries.prior_db_conn_obj. \
            fetch_entity_by_mention_size_emnlp17(cur_m, size=0)

    # XL: find from YAGO
    if not can_ents_cur_m:
        cur_m = normalize_form(m)
        can_ents_cur_m = Dictionaries.prior_db_conn_obj.fetch_entity_by_mention_size(cur_m, size=0)
    return cur_m, can_ents_cur_m


if __name__ == '__main__':
    log.info("Starting system...")
    task_scheduler.constant.init_dictionaries()
    # log.info("Seed Entity: {}".format(Dictionaries.seed_entity))

    import sys
    if len(sys.argv) > 1:
        data_source = sys.argv[1]
    else:
        raise ValueError('Data source not specified')
    can_size = 50
    if len(sys.argv) > 2:
        can_size = int(sys.argv[2])

    print('Generating tasks for data source: %s' % data_source)
    print('Candidate size: %d' % can_size)
    if data_source == 'aida_conll':
        generate_tasks(can_size)
    elif data_source == 'msnbc':
        generate_tasks_msnbc(can_size)
    elif data_source == 'ace':
        generate_tasks_ace(can_size)
    elif data_source == 'aquaint':
        generate_tasks_aquaint(can_size)
    elif data_source == 'wiki_uiuc':
        generate_tasks_wiki_uiuc(can_size)
    elif data_source == 'clueweb':
        generate_tasks_clueweb(can_size)
    elif data_source == 'wiki_wned':
        generate_tasks_wiki_wned(can_size)
    elif data_source == 'kore':
        generate_tasks_kore(can_size)
    elif data_source == 'rss':
        generate_tasks_rss(can_size)
    else:
        raise ValueError('invalid data source: %s' % data_source)

    # generate_tasks_sup()
    # generate_tasks_msnbc()
    #generate_tasks_ace()
    # generate_tasks_aquaint()
    # generate_tasks_kore()
    # generate_tasks_wiki_uiuc()
    # while entity_sets:
    #     log.info("Total tasks: {}".format(len(entity_sets)))
    #     entity_sets = generate_tasks(entity_sets)
    # break
    Dictionaries.db.close_db_conn()

    # print(tuple(ids))
