I'll start by checking the actual signature of `funding_rates` in the SDK adapter, since the error mirrors the past diagnosis pattern (a hallucinated kwarg).

¡

¡

¡

¡

¡

¡

Let me inspect the SDK adapter.

¡

¡

¡

¡

¡

¡


<invoke name="Bash">
<parameter name="command">grep -n "def funding_rates" /root/crucible/sdk/adapters.py</parameter>
</invoke>


¡

¡

¡

¡

¡

¡

102:def funding_rates(