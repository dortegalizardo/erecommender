# Generated by Django 3.2.12 on 2022-03-28 10:00

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('recommender', '0004_auto_20220328_0950'),
    ]

    operations = [
        migrations.AddField(
            model_name='workflow',
            name='processing_times',
            field=models.JSONField(blank=True, default=dict, help_text='Field saved for keeping track of time it take to process data.'),
        ),
    ]
