"""Configurable student model; initialize independently from teacher weights."""
from .model import Model

class EchoRenderStudent(Model):
    pass

Student = EchoRenderStudent
