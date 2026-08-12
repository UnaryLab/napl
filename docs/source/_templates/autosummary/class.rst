{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. The autosummary ``attributes`` context list also holds properties, so the
   metadata attributes below are named one by one rather than filtered from it.
{% set metadata = ['hw', 'streaming', 'internal_encode', 'encoding_io', 'polarity_io', 'correlation_i', 'flux_stability'] %}
.. autoclass:: {{ objname }}
   :members:
   :member-order: bysource
   :exclude-members: __init__, __new__{% if objname != 'napl_base' %}{% for name in metadata if name != objname %}, {{ name }}{% endfor %}{% endif %}

   .. raw:: html

      <hr class="api-member-divider">

   .. automethod:: __init__
{% if 'reset' in inherited_members %}

   .. automethod:: reset
{% endif %}
