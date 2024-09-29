import datetime

from marshmallow import Schema, fields, post_load, EXCLUDE


class EpochDateTime(fields.Integer):
    """Use Epoch time for serialized and DateTime deserialized data."""

    def _serialize(self, value, attr, obj, **kwargs):
        """Convert datetime to timestamp for external use."""
        if value is None:
            return None
        try:
            value = value.timestamp() * 1e3  # convert from `s` to `ms`
            return super()._serialize(value, attr, obj, **kwargs)
        except TypeError:
            raise TypeError(f'Expected datetime not "{type(value)}"')

    def _deserialize(self, value, attr, data, **kwargs):
        """Convert timestamp to datetime for internal use.

        :param value: timestamp in milliseconds
        """
        value = super()._deserialize(value, attr, data, **kwargs)
        try:
            return datetime.datetime.fromtimestamp(
                value / 1e3
            )  # convert from `ms` to `s`
        except TypeError:
            raise TypeError(f'Expected timestamp not "{type(value)}"')


class RoundedFloat(fields.Field):
    """Add rounding to the Float field."""

    def __init__(self, places=2, *args, **kwargs):
        self.places = places
        super().__init__(*args, **kwargs)

    def _serialize(self, value, attr, obj, **kwargs):
        if value is None:
            return None
        return round(value, self.places)


def snake_to_camel(val):
    components = val.split("_")
    return "".join([components[0].lower()] + [item.title() for item in components[1:]])


class BaseSchema(Schema):
    """Add functionality expected across all schemas."""

    def __init__(self, *args, **kwargs):
        """Serialize/Deserialize fields to/from camelCase.

        https://marshmallow.readthedocs.io/en/stable/quickstart.html#specifying-serialization-deserialization-keys
        """
        super().__init__(*args, **kwargs)
        for field_name, field in self.fields.items():
            if not field.data_key:
                # Allows implementor to override default behaviour
                field.data_key = snake_to_camel(field_name)

    class Meta:
        unknown = EXCLUDE


class UpdateSchemaMixin:
    """Add update related fields."""

    updated = EpochDateTime()


class PaginateSchemaMixin:
    limit = fields.Integer(
        description="Number of records to return, defaults to 50, maximum is 50.",
        example=10,
        missing=100,
    )
    offset = fields.Integer(
        description="Offset number of records, defaults to 0.", example=150, missing=0
    )

    @post_load
    def post_load(self, data, **kwargs):
        if data["limit"] > 100:
            data["limit"] = 100
        return data
