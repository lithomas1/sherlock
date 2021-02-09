<script>
  import InputDemo from './InputDemo.svelte';
  import TwitterDemo from './TwitterDemo.svelte';

  let modeStatus = 'deciding';

  const chooseInput = () => modeStatus = 'input';
  const chooseTwitter = () => modeStatus = 'twitter';
  const home = () => modeStatus = 'deciding';
  const keyboard = 'https://images.vexels.com/media/users/3/153083/isolated/preview/9f5fcfd398a9d9c52a586cbe1d67789c-computer-keyboard-stroke-icon-by-vexels.png';
  const twitter = 'https://static.adweek.com/adweek.com-prod/wp-content/uploads/2017/04/perspective-twitter-head-content-2017.png';
</script>

<main>
  <h1 on:click={home}>Sherlock demo</h1>
  {#if modeStatus === 'deciding'}
    <div class="button-container">
      <div class="input-button not-phony" on:click={chooseInput}>
        <img alt="select keyboard demo" src={keyboard} />
      </div>
      <div class="separator-button" />
      <div class="twitter-button not-phony" on:click={chooseTwitter}>
        <img alt="select twitter demo" src={twitter} />
      </div>
    </div>
  {:else if modeStatus === 'input'}
    <InputDemo />
  {:else if modeStatus === 'twitter'}
    <TwitterDemo />
  {/if}
</main>

<style>
  main {
    text-align: center;
    padding: 1em;
    max-width: 240px;
    margin: 0 auto;
    height: 100%;
  }

  h1 {
    color: #ff3e00;
    font-size: 4em;
    font-weight: 100;
  }

  h1:hover {
    cursor: pointer;
  }

  img {
    object-fit: cover;
    height: 40vh;
    width: 100%;
  }

  .button-container {
    margin: auto;
    padding: auto;
    width: 80vw;
    max-height: 40vh;
    display: flex;
    flex-direction: row;
  }

  .input-button {
    background-color: red;
    height: 100%;
    flex-grow: 1;
  }

  .twitter-button {
    height: 100%;
    flex-grow: 1;
  }

  .separator-button {
    height: 100%;
    flex-grow: .15;
  }

  .not-phony {
    padding: 0px;
    transition: .2s ease-in-out;
    max-width: 40vw;
  }

  .not-phony:hover {
    cursor: pointer;
    transform: scale(1.05);
  }

  @media (min-width: 640px) {
    main {
      max-width: none;
    }
  }
</style>
