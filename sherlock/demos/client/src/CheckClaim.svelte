<script>
  export let claim = '';

  let agree = [];
  let disagree = [];

  async function getVerification(claim) {
    return await fetch(`/verify?claim=${claim}`)
      .then(r => r.json());
  }

  $: r = (async (claim) => {
    const verification = await getVerification(claim);
    agree = verification.agree;
    disagree = verification.disagree;
  })(claim);

</script>

<div class="check-claim">
  <div class="agree">
    <table>
      <tr>
        <th>Agree</th>
        <th>Confidence</th>
      </tr>
      {#each agree as ag}
        <tr>
          <td>{ag[0]}</td>
          <td>{ag[1]}</td>
        </tr>
      {/each}
    </table>
  </div>
  <div class="disagree">
    <table>
      <tr>
        <th>Disagree</th>
        <th>Confidence</th>
      </tr>
      {#each disagree as dg}
        <tr>
          <td>{dg[0]}</td>
          <td>{dg[1]}</td>
        </tr>
      {/each}
    </table>
  </div>
</div>

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
