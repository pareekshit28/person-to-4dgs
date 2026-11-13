import { readFTGS } from "./ftgs.js";
import { MAX_AUDIO_BYTES } from "./playback.js";
import { openZip } from "./zip.js";

/** Detect packaged scenes by their contents so File and Blob loads agree. */
export async function readModel(blob, options = {}) {
  options.signal?.throwIfAborted();
  const signature = new DataView(await blob.slice(0, 4).arrayBuffer());
  if (
    signature.byteLength === 4 &&
    signature.getUint32(0, true) === 0x04034b50
  ) {
    const archive = await openZip(blob, options.signal);
    if (!archive.names.includes("meta.json") && archive.names.includes("character.ftgs.ply")) {
      const metadata = JSON.parse(new TextDecoder().decode(await archive.read("playback.json", 16 * 1024)));
      if (!Number.isFinite(metadata?.fps) || metadata.fps <= 0 || metadata.audio !== "audio.m4a")
        throw new Error("Invalid Person4D playback metadata.");
      const audio = await archive.read("audio.m4a", MAX_AUDIO_BYTES);
      if (!audio.length) throw new Error("The scene bundle contains an empty audio track.");
      const model = await archive.storedBlob("character.ftgs.ply", 512 * 1024 * 1024);
      return {
        ...(await readFTGS(model, options)),
        format: "ftgs-ply",
        timelineMode: 0,
        playback: { fps: metadata.fps },
        audio: { blob: new Blob([audio], { type: "audio/mp4" }) },
      };
    }
    const { readTSOG } = await import("./tsog.js");
    return readTSOG(blob, { compactHarmonics: true, ...options });
  }
  return {
    ...(await readFTGS(blob, options)),
    format: "ftgs-ply",
    timelineMode: 0,
  };
}
