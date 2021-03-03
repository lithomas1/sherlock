<script>
  export let claim = '';

  let agree = [];
  let disagree = [];
  let error = '';

  async function getVerification(claim) {
    return await fetch(`/verify?claim=${claim}`)
      .then(r => r.json());
  }

  $: r = (async (claim) => {
    try {
      const verification = await getVerification(claim);
      agree = verification.agree;
      disagree = verification.disagree;
      error = '';
    } catch (e_) {
      error = 'Request failed';
    }
  })(claim);

</script>

{#if error}
  <p>Looks like something went wrong</p>
  <p><code>{error}</code></p>
{:else}
  <div class="check-claim">
    <div class="agree">
      <table>
        <tr>
          <th>Agree</th>
          <th>Confidence</th>
          <th>Not significant</th>
        </tr>
        {#each agree as ag}
          <tr>
            <td>{ag[0]}</td>
            <td>{ag[1]}</td>
            <td>{ag[2]}</td>
          </tr>
        {/each}
      </table>
    </div>
    <div class="disagree">
      <table>
        <tr>
          <th>Disagree</th>
          <th>Confidence</th>
          <th>Not significant</th>
        </tr>
        {#each disagree as dg}
          <tr>
            <td>{dg[0]}</td>
            <td>{dg[1]}</td>
            <td>{dg[2]}</td>
          </tr>
        {/each}
      </table>
    </div>
  </div>
{/if}

<style>
  table {
    border-collapse: collapse;
  }

  .check-claim {
    margin: auto;
    display: flex;
    flex-direction: row;
  }

  tr:nth-child(even) {
    background: #DDDDDD;
  }

  td, th {
    border: 1px solid #000000;
    text-align: center;
    padding: 8px;
  }

  .agree, .disagree {
    margin: auto;
  }

  .agree {
    margin-right: 2.5%;
  }

  .disagree {
    margin-left: 2.5%;
  }
</style>
