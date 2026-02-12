from tortoise import fields
from tortoise.models import Model


class Battle(Model):
    id = fields.IntField(pk=True)
    model_a_name = fields.CharField(max_length=128)
    model_b_name = fields.CharField(max_length=128)
    start_word = fields.CharField(max_length=32)
    history = fields.JSONField(default=list)
    winner = fields.CharField(max_length=8)  # "A", "B", "draw"
    reason = fields.CharField(max_length=128)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta:
        table = "battles"
        ordering = ["-created_at"]
