from .common import * # noqa

DEBUG = True

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'erecommend',
        'USER': 'elibro',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}