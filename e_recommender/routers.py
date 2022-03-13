class AnalyticsRouter:
    """
    A router to control all database operations on models in the
    analytics application.
    """

    def db_for_read(self, model, **hints):
        """
        Attempts to read analytics models go to analytics.
        """
        if model._meta.app_label == 'analytics':
            return 'analytics_replica'
        return None

    def db_for_write(self, model, **hints):
        """
        Attempts to write analytics models go to analytics.
        """
        if model._meta.app_label == 'analytics':
            return 'analytics'
        return None

    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if a model in the analytics app is involved.
        """
        if obj1._meta.app_label == 'analytics' or \
           obj2._meta.app_label == 'analytics':
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Make sure the analytics app only appears in the 'analytics'
        database.
        """
        if app_label == 'analytics' and hints.get('target_db') != 'default':
            return db == 'analytics'
        return None


class DefaultRouter:
    """
    Default Router.
    """

    def db_for_read(self, model, **hints):
        """
        Attempts to read content models go to content.
        """
        if model._meta.app_label == 'content':
            return 'default'
        return None

    def db_for_write(self, model, **hints):
        """
        Attempts to write content models go to content.
        """
        if model._meta.app_label == 'content':
            return 'default'
        return None

    def allow_relation(self, obj1, obj2, **hints):
        """
        Allow relations if a model in the content app is involved.
        """
        if obj1._meta.app_label == 'contents' and \
           obj2._meta.app_label == 'content':
            return True
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        """
        Make sure the content app only appears in the 'content'
        database.
        """
        if app_label == 'content' or hints.get('target_db') == 'default':
            return db == 'default'
        return None
