import task_scheduler.constant
from task_scheduler import logging_config
import task_scheduler.database.db_psql

log = logging_config.getLogger()


def get_tasks():
    logging_config.info("Getting new task...")
    task_scheduler.constant.Dictionaries.db = task_scheduler.database.db_psql.PostgreSQLDB()
    db = task_scheduler.constant.Dictionaries.db
    tasks = db.fetch_tasks_done()
    if tasks is None:
        return tasks
    else:
        entities = [task[0] for task in tasks]
    return entities


def update_task(wiki_title):
    logging_config.info("Updating task: {}...".format(wiki_title))
    db = task_scheduler.constant.Dictionaries.db
    db.update_tasks_queue(wiki_title)

