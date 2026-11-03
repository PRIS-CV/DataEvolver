export async function api<T = unknown>(
  path: string,
  value?: unknown,
): Promise<T> {
  const response = await fetch(
    "/api/" + path,
    value === undefined
      ? {}
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(value),
        },
  );
  if (response.status === 401)
    throw new Error(
      "Open the startup URL printed by apps/workbench/launch.sh / 请打开 apps/workbench/launch.sh 输出的登录链接",
    );
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      (data.error || response.statusText) +
        (data.receipt ? ` · receipt: ${data.receipt}` : ""),
    );
  return data;
}
