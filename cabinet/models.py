from django.db import models


class BtcPrice(models.Model):
    time = models.DateTimeField()
    price = models.FloatField()
