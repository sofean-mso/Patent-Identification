# Copyright 2024 FIZ-Karlsruhe (Mustafa Sofean)

import os

import psycopg2 as postgres
from configparser import ConfigParser
from configparser import ConfigParser

PROJECT_ROOT = os.path.dirname(os.path.dirname( os.path.abspath(__file__)))


class PostgresUtils:
    """
    Utilization of postgres database
    """
    def __init__(self):
        """
        initialize the database
        """
        db_params = self.get_config()
        self.conn = postgres.connect(**db_params)

    def get_config(self, filename=PROJECT_ROOT+'/configuration.ini', section='postgresql'):
        """
        get postgres configs
        :param filename:
        :param section:
        :return:
        """
        # create a parser
        parser = ConfigParser()
        # read config file
        parser.read(filename)

        # get section, default to postgresql
        db_params = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db_params[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, filename))

        return db_params

    def get_doc_ids(self):
        """
        Get all ids in the postgres zable
        :return:
        """
        try:
            doc_ids = set()
            with self.conn.cursor() as cursor:
                cursor.execute("SELECT ID FROM mvp_patents.A61K0035_16_segs")
                while True:
                    rows = cursor.fetchmany(5000)
                    if not rows:
                        break
                    for (id,) in rows:
                        doc_ids.add(id)
        except (Exception, postgres.DatabaseError) as error:
            print(error)

        return doc_ids

    def get_doc_data_by_id(self, doc_id:str):
        """
        Get the document data for related ID
        :param doc_id:
        :return:
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT DATA FROM mvp_patents.A61K0035_16_segs  where ID='" + doc_id + "'")
            doc_data = cursor.fetchone()[0]

        except (Exception, postgres.DatabaseError) as error:
            print(error)

        return doc_data
