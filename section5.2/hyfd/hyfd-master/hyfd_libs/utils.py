import os
import json
import datetime
import time

STAT_DIRECTORY = './results/'
STAT_FILE = 'hyfd_results.txt'

class Stats(object):
    def __init__(self, logger, headers, restart=False):
        self.logger = logger
        self.headers = headers
        if not os.path.isdir(STAT_DIRECTORY):
            try:
                os.mkdir(STAT_DIRECTORY)
                self.logger.info("Directory Created: {}".format(STAT_DIRECTORY))
            except:
                self.logger.error("Director does not exists: {}".format(STAT_DIRECTORY))
                self.logger.error("EXITING: Could not create directory: {}".format(STAT_DIRECTORY))
                exit()
        if not os.path.isfile(STAT_DIRECTORY+STAT_FILE) or restart:
            with open(STAT_DIRECTORY+STAT_FILE, 'w') as fout:
                self.logger.info("Results File Initialized: {}".format(STAT_DIRECTORY+STAT_FILE))
                fout.write('{}\n'.format('\t'.join(self.headers)))

    def log_results(self, results):
        with open(STAT_DIRECTORY+STAT_FILE, 'a') as fout:
            fout.write('{}\n'.format('\t'.join(results)))

OUTPUT_DIRECTORY = './json/'
OUTPUT_FNAME = '{}-{}.json'

class Output(object):
    def __init__(self, logger, db_path):
        self.logger = logger
        self.dbname = db_path[db_path.rfind('/')+1:db_path.rfind('.')]
        self.st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')
        
        
        if not os.path.isdir(OUTPUT_DIRECTORY):
            try:
                os.mkdir(OUTPUT_DIRECTORY)
                self.logger.info("Directory Created: {}".format(OUTPUT_DIRECTORY))
            except:
                self.logger.error("Director does not exists: {}".format(OUTPUT_DIRECTORY))
                self.logger.error("EXITING: Could not create directory: {}".format(OUTPUT_DIRECTORY))
                exit()
        self.fout_path = OUTPUT_DIRECTORY+OUTPUT_FNAME.format(self.dbname, self.st)

    def write(self, fds):
        with open(self.fout_path, 'w') as fout:
            json.dump(list(fds), fout)
            self.logger.info("FDs written in: {}".format(self.fout_path))