from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)

class heartrate_model(models.Model):

    names=models.CharField(max_length=300)
    VLF=models.CharField(max_length=300)
    VLF_PCT=models.CharField(max_length=300)
    LF=models.CharField(max_length=300)
    LF_PCT=models.CharField(max_length=300)
    LF_NU=models.CharField(max_length=300)
    HF=models.CharField(max_length=300)
    HF_PCT=models.CharField(max_length=300)
    HF_NU=models.CharField(max_length=300)
    TP=models.CharField(max_length=300)
    LF_HF=models.CharField(max_length=300)
    HF_LF=models.CharField(max_length=300)


class hrpredict_model(models.Model):

    names=models.CharField(max_length=300)
    VLF=models.CharField(max_length=300)
    VLF_PCT=models.CharField(max_length=300)
    LF=models.CharField(max_length=300)
    LF_PCT=models.CharField(max_length=300)
    LF_NU=models.CharField(max_length=300)
    HF=models.CharField(max_length=300)
    HF_PCT=models.CharField(max_length=300)
    HF_NU=models.CharField(max_length=300)
    TP=models.CharField(max_length=300)
    LF_HF=models.CharField(max_length=300)
    HF_LF=models.CharField(max_length=300)
    hrpredict = models.CharField(max_length=300)

class Stress_Accuracy_model(models.Model):

    names = models.CharField(max_length=300)
    acc = models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)