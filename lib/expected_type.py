import re


class ExpectedType:
    def check(self, d: dict):
        pass

    def describe(self):
        pass


class BasicExpectedType(ExpectedType):
    def __init__(self, key, data_type):
        self.key = key
        if isinstance(data_type, (tuple, list)):
            self.data_types = data_type
        else:
            self.data_types = [data_type]

    def check(self, d) -> str or None:
        v = d.get(self.key)
        if v is None:
            return self.key + " missing, " + self.describe()

        matched_allowed_type = False
        for allowed_type in self.data_types:
            if isinstance(v, allowed_type):
                matched_allowed_type = True

        if matched_allowed_type:
            return None
        else:
            return self.key + " error, " + self.describe()

    def describe(self):
        return "type " + "/".join(map(lambda t: t.__name__, self.data_types))


class BasicExpectedRegex(BasicExpectedType):
    def __init__(self, key, regex_str):
        data_type = str
        BasicExpectedType.__init__(self, key, data_type)
        self.regex_str = regex_str

    def check(self, d: dict):
        error = super().check(d)

        if error is not None:
            return error

        v = d.get(self.key)
        pattern = re.compile(self.regex_str)
        if not pattern.match(v):
            return self.key + " error, " + self.describe()

    def describe(self):
        return str(super().describe()) + f" (Must satisfy regex '{self.regex_str}')"


class condition_ops:
    AND = "AND"
    OR = "OR"
    XOR = "XOR"


class Condition:
    def __init__(
        self,
        condition1=None,
        operation=condition_ops.AND,
        condition2=None,
        key: str = None,
        value=None,
    ) -> None:
        self.condition1: Condition = condition1
        self.condition2: Condition = condition2
        self.operation = operation
        self.key = key
        self.value = value

        if operation not in condition_ops.__dict__:
            raise Exception("Invalid operation in conditions")

    def check(self, d: dict) -> bool:

        if self.key is not None and self.value is not None:
            return d.get(self.key) == self.value

        if self.condition1:
            result1 = self.condition1.check(d)
        else:
            result1 = True

        if self.condition2:
            result2 = self.condition2.check(d)
        else:
            result2 = True

        if self.operation == condition_ops.AND:
            return result1 and result2

        if self.operation == condition_ops.OR:
            return result1 or result2

        if self.operation == condition_ops.XOR:
            return result1 + result2 == 1


class ConditionalMandatory(BasicExpectedType):
    def __init__(
        self,
        field: BasicExpectedType,
        key: str = None,
        value=None,
        condition: Condition = None,
    ) -> None:
        """Field is only mandatory if condition is met"""

        if not condition:
            if value is None or key is None:
                raise Exception(
                    "Either Key Val pair or Conditions tuple must be supplied"
                )
            condition = Condition(key=key, value=value)

        self.condition = condition
        self.field_base = field
        self.key = field.key
        self.data_types = field.data_types

    def check(self, d: dict):
        v = d.get(self.key)
        mandatory = self.condition.check(d)
        if v is None and not mandatory:
            return None

        return self.field_base.check(d)

    def describe(self):
        return self.field_base.describe()


class Optional(BasicExpectedType):
    def __init__(
        self,
        field: BasicExpectedType,
    ) -> None:
        """Field is only mandatory if condition is met"""

        self.field_base = field
        self.key = field.key
        self.data_types = field.data_types

    def check(self, d: dict):
        v = d.get(self.key)
        if v is None:
            return None

        return self.field_base.check(d)

    def describe(self):
        return self.field_base.describe()


class SetDefault(BasicExpectedType):
    def __init__(self, field: BasicExpectedType, default_value) -> None:
        """Sets default value if field is missing"""
        self.default_value = default_value
        self.field_base = field
        self.key = field.key
        self.data_types = field.data_types

    def check(self, d: dict):
        v = d.get(self.key)
        if v is None:
            d[self.key] = self.default_value

        return self.field_base.check(d)

    def describe(self):
        return self.field_base.describe()


class ConstrainedExpectedType(BasicExpectedType):
    def __init__(self, key, data_types, allowed_values):
        BasicExpectedType.__init__(self, key, data_types)
        self.allowed_values = allowed_values

    def check(self, d):
        error = super().check(d)

        if error is not None:
            return error
        else:
            v = d.get(self.key)
            for allowed_value in self.allowed_values:
                if v == allowed_value:
                    return None

        return self.key + " error, " + self.describe()

    def describe(self):
        return super().describe() + " (" + " or ".join(self.allowed_values) + ")"


class AlternativeExpectedType(ExpectedType):
    def __init__(self, alternatives: tuple):
        self.alternatives = alternatives

    def check(self, v):
        errors = []
        for alternative in self.alternatives:
            error = alternative.check(v)
            if error is None:
                return None
            else:
                errors.append(error)

        return " || ".join(errors)

    def describe(self):
        return " || ".join(
            map(lambda a: a[0] + "[" + a[1].describe() + "]", self.alternatives.items())
        )


def check_dict_against_fields(d: dict, fields: tuple):
    errors = []

    for expected_type in fields:
        error = expected_type.check(d)
        if error is not None:
            errors.append(error)

    if len(errors) == 0:
        return None
    else:
        return "\n".join(errors)


def check_dict_additional_fields(d: dict, fields: tuple):
    expected_fields = [field.key for field in fields]
    additonal_fields = []

    for item in d:
        if item not in expected_fields:
            additonal_fields.append(item)

    if len(additonal_fields) == 0:
        return None

    return additonal_fields
