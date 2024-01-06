{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}


{% block attributes %}
{% if attributes %}
.. rubric:: Module Attributes
.. autosummary::
   :toctree:
{% for item in attributes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}


{% block functions %}
{% if functions %}
.. rubric:: Functions
.. autosummary::
   :toctree:
{% for item in functions %}
   {{ item }}
{%- endfor %}
   {% endif %}
{% endblock %}


{% block all_classes %}
{% if all_classes %}
.. rubric:: Classes
.. autosummary::
   :toctree:
   :template: class.rst
{% for item in all_classes %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}


{% block exceptions %}
{% if exceptions %}
.. rubric:: Exceptions
.. autosummary::
   :toctree:
{% for item in exceptions %}
   {{ item }}
{%- endfor %}
{% endif %}
{% endblock %}


{% block all_modules %}
{% if all_modules %}
.. currentmodule:: {{ fullname }}
.. rubric:: Modules
.. autosummary::
   :toctree:
   :template: module.rst
   :recursive:
{% for item in all_modules %}
{% if not item.endswith("tests") %}
   {{ item.split(".")[-1] }}
{% endif %}
{%- endfor %}
{% endif %}
{% endblock %}
