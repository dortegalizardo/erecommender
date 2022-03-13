# Generated by Django 3.2.12 on 2022-03-06 03:20

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Recommendation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('order', models.IntegerField(default=0)),
                ('average_rating', models.DecimalField(decimal_places=2, help_text="This is a calculated field, please don't modify", max_digits=6, verbose_name='Rating')),
            ],
            options={
                'verbose_name': 'Recommendation',
                'verbose_name_plural': 'Recommendations',
            },
        ),
        migrations.CreateModel(
            name='Title',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('identifier', models.CharField(help_text='Input the book identifier.', max_length=20)),
                ('publisher', models.CharField(blank=True, help_text='Input the publisher associated to the book.', max_length=120, verbose_name='Publisher')),
                ('theme', models.CharField(help_text='Input the theme associated to the book.', max_length=100, verbose_name='Theme')),
                ('name', models.CharField(blank=True, help_text='Input the name given to the resource.', max_length=250, verbose_name='Title name')),
                ('complete_text', models.TextField(help_text='Complete text of the book.', verbose_name='Raw Text')),
                ('parsed_tokens', models.TextField(help_text='Input tokens separated by a comma', verbose_name='Tokens')),
            ],
            options={
                'verbose_name': 'Ranked Title',
                'verbose_name_plural': 'Ranked Titles',
            },
        ),
        migrations.CreateModel(
            name='RecommendationRating',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('rating', models.IntegerField(default=0)),
                ('recommendation', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='recommender.recommendation')),
            ],
            options={
                'verbose_name': 'Title Rating',
                'verbose_name_plural': 'Title Ratings',
            },
        ),
        migrations.AddField(
            model_name='recommendation',
            name='recommendation',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='recommendation', to='recommender.title', verbose_name='Recommendation'),
        ),
        migrations.AddField(
            model_name='recommendation',
            name='title',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='base_title', to='recommender.title', verbose_name='Current Title'),
        ),
    ]